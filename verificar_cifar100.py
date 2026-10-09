"""Audita los checkpoints con un evaluador y un control de aprendizaje independientes."""

import os

# Con CIFAR-100 dentro de la imagen Docker se usa la copia local, sin Internet.
if os.environ.get("CFL_DATA_PRELOADED") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from torchvision.models import resnet18
from cifar100_sampling import balanced_subset, balanced_order
from cifar100_validation import learning_problem
import math


CIFAR100_MEAN = torch.tensor((0.5071, 0.4865, 0.4409)).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor((0.2673, 0.2564, 0.2762)).view(1, 3, 1, 1)


def tensors(dataset, normalize=False):
    """Reconstruye los tensores sin usar el código del experimento.

    ``normalize`` reproduce la normalización de CIFAR-100 que aplican por
    defecto las ejecuciones nuevas (``config.json`` → ``normalize``); las
    ejecuciones antiguas no la tienen.
    """
    images = np.stack([np.asarray(row["img"]) for row in dataset])
    labels = [int(row["fine_label"]) for row in dataset]
    x = (torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255).contiguous()
    if normalize:
        x = (x - CIFAR100_MEAN) / CIFAR100_STD
    return x, torch.tensor(labels)


def image_hashes(dataset):
    return {hashlib.sha256(np.asarray(row["img"]).tobytes()).hexdigest() for row in dataset}


def image_records(dataset):
    return Counter((hashlib.sha256(np.asarray(row["img"]).tobytes()).hexdigest(), int(row["fine_label"]))
                   for row in dataset)


def new_model():
    model = resnet18(weights=None, num_classes=100)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def weight_aligned(model):
    """Weight Aligning reimplementado aparte: iguala la norma media de las filas B (50-99) a la de A (0-49)."""
    with torch.no_grad():
        weight, bias = model.fc.weight, model.fc.bias
        gamma = weight[:50].norm(dim=1).mean() / weight[50:].norm(dim=1).mean()
        weight[50:] *= gamma
        bias[50:] *= gamma
    return model


def evaluate(model, images, labels, keep_indices=None):
    model.eval()
    predictions = []
    group_predictions = []
    loss_sum = 0.0
    offset = 0
    with torch.inference_mode():
        for batch in images.split(32):
            logits = model(batch)
            assert torch.isfinite(logits).all(), "La evaluación contiene logits no finitos"
            targets_batch = labels[offset:offset + len(batch)]
            predictions.extend(logits.argmax(dim=1).tolist())
            group_predictions.extend(torch.where(targets_batch < 50, logits[:, :50].argmax(dim=1),
                                                  logits[:, 50:].argmax(dim=1) + 50).tolist())
            loss_sum += torch.nn.functional.cross_entropy(logits, targets_batch, reduction="sum").item()
            offset += len(batch)
    targets = labels.tolist()
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    result = {
        "correct": correct, "total": len(targets), "accuracy_percent": 100 * correct / len(targets),
        "prediction_counts": dict(Counter(predictions)),
        "predictions_group_a": sum(pred < 50 for pred in predictions),
        "predictions_group_b": sum(pred >= 50 for pred in predictions),
        "mean_cross_entropy": loss_sum / len(targets),
        "diagnostic_accuracy_with_true_group_given_percent":
            100 * sum(pred == target for pred, target in zip(group_predictions, targets)) / len(targets),
    }
    if keep_indices is not None:
        kept_correct = sum(predictions[index] == targets[index] for index in keep_indices)
        result["without_exact_train_duplicates"] = {
            "correct": kept_correct, "total": len(keep_indices),
            "accuracy_percent": 100 * kept_correct / len(keep_indices),
            "excluded_samples": len(targets) - len(keep_indices),
        }
    return result


def main():
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    config = json.loads((args.audit_dir / "config.json").read_text())
    audit = json.loads((args.audit_dir / "audit.json").read_text())
    assert config["device"] == "cpu", "Este verificador recalcula los checkpoints en CPU"
    buffers = config.get("buffers", [0, 5, 10, 15, 20, 25])
    metrics_path = args.audit_dir / "results.csv"
    if not metrics_path.exists():
        metrics_path = args.audit_dir / "diagnostic_results.csv"
    if not metrics_path.exists() and config.get("validation"):
        status = json.loads((args.audit_dir / "validation.json").read_text())
        if not status["comparison_valid"]:
            print(f"COMPARACIÓN NO DISPONIBLE: {status['reason']}", flush=True)
            raise SystemExit(2)
    torch.set_num_threads(4)
    normalize = bool(config.get("normalize", False))

    test = load_dataset("uoft-cs/cifar100", revision=config["dataset_revision"], split="test")
    # Flower 0.6.1 baraja TODOS los splits antes de filtrar o particionar.
    test = test.shuffle(seed=config["seed"])
    official_train = load_dataset("uoft-cs/cifar100", revision=config["dataset_revision"], split="train")
    official_train_records = image_records(official_train)
    official_train_hashes = {digest for digest, label in official_train_records}
    test_groups = {}
    keep_indices_by_group = {}
    result = {"torch_version": torch.__version__, "test_groups": {}, "evaluations": {}, "checks": {}}
    for name, lower, upper in (("A", 0, 50), ("B", 50, 100)):
        group = test.filter(lambda row: lower <= row["fine_label"] < upper)
        if config.get("test_samples_per_group") is None:
            pass
        elif config.get("sampling") == "balanced":
            group = balanced_subset(group, config["test_samples_per_group"], config["seed"])
        else:
            group = group.shuffle(seed=config["seed"]).select(range(config["test_samples_per_group"]))
        test_groups[name] = group
        keep_indices_by_group[name] = [index for index, row in enumerate(group)
                                      if hashlib.sha256(np.asarray(row["img"]).tobytes()).hexdigest()
                                      not in official_train_hashes]
        x, y = tensors(group, normalize)
        result["test_groups"][name] = {
            "samples": len(group), "classes_present": len(set(y.tolist())),
            "label_counts": dict(Counter(y.tolist())), "image_shape": list(x.shape),
            "pixel_range": [x.min().item(), x.max().item()],
            "exact_duplicates_in_official_train": len(group) - len(keep_indices_by_group[name]),
        }
        expected_counts = audit["datasets"][0 if name == "A" else 1]["label_counts"]
        assert {str(key): value for key, value in Counter(y.tolist()).items()} == expected_counts

    with metrics_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    completed_buffers = [0 if row["fraction"] == "No Replay" else int(row["fraction"].split("%")[0]) for row in rows]
    assert completed_buffers and all(percent in buffers for percent in completed_buffers)
    if metrics_path.name == "results.csv":
        assert completed_buffers == buffers
    buffers = completed_buffers
    tensors_by_group = {name: tensors(group, normalize) for name, group in test_groups.items()}

    for phase, filename in [("phase1", "audit_phase1.pt")] + [
        (f"phase2_{pct}", f"audit_phase2_{pct}.pt") for pct in buffers
    ]:
        model = new_model()
        model.load_state_dict(torch.load(args.audit_dir / filename, map_location="cpu", weights_only=True))
        result["evaluations"][phase] = {
            name: evaluate(model, x, y, keep_indices_by_group[name])
            for name, (x, y) in tensors_by_group.items()
        }
        print(phase, json.dumps(result["evaluations"][phase]), flush=True)

    base = result["evaluations"]["phase1"]["A"]["accuracy_percent"]
    for row, pct in zip(rows, buffers):
        metrics = result["evaluations"][f"phase2_{pct}"]
        expected = {"step3_acc_a": base, "step4_acc_a": metrics["A"]["accuracy_percent"],
                    "step4_acc_b": metrics["B"]["accuracy_percent"],
                    "loss_a": base - metrics["A"]["accuracy_percent"]}
        for key, value in expected.items():
            assert abs(float(row[key]) - value) < 1e-10, (pct, key, row[key], value)
    result["checks"]["all_csv_metrics_match_independent_evaluation"] = True
    if rows and rows[0].get("wa_acc_a") not in (None, ""):
        for row, pct in zip(rows, buffers):
            model = new_model()
            model.load_state_dict(torch.load(args.audit_dir / f"audit_phase2_{pct}.pt", map_location="cpu", weights_only=True))
            weight_aligned(model)
            for name, key in (("A", "wa_acc_a"), ("B", "wa_acc_b")):
                value = evaluate(model, *tensors_by_group[name])["accuracy_percent"]
                assert abs(float(row[key]) - value) < 1e-10, (pct, key, row[key], value)
        result["checks"]["weight_aligning_metrics_match_independent_evaluation"] = True
    initial_weights = audit["phase2_initial_weights"]
    assert len(initial_weights) == len(buffers) and all(
        value == audit["phase1_weights_sha256"] for value in initial_weights.values()
    )
    result["checks"]["all_buffers_start_from_same_weights"] = True
    expected_calls = config["num_clients"] * config["num_rounds"] * (len(buffers) + (0 if config.get("phase1_checkpoint") else 1))
    assert len(audit["training"]) == expected_calls
    result["training_exposure"] = {}
    for percent in buffers:
        calls = [row for row in audit["training"]
                 if row["context"]["phase"] == 2
                 and round(row["context"]["replay_fraction"] * 100) == percent]
        observed = {(row["context"]["round"], row["context"]["client"]) for row in calls}
        expected = {(r, client) for r in range(1, config["num_rounds"] + 1)
                    for client in range(config["num_clients"])}
        assert len(calls) == len(expected) and observed == expected
        totals = Counter()
        for row in calls:
            labels = Counter({int(label): count for label, count in row["label_counts"].items()})
            assert all(0 <= label < 100 and count > 0 for label, count in labels.items())
            assert sum(labels.values()) == row["samples_processed"]
            assert len(row["batch_losses"]) == row["optimizer_steps"]
            assert all(math.isfinite(loss) for loss in row["batch_losses"])
            a_seen = sum(count for label, count in labels.items() if label < 50)
            assert (a_seen == 0) if percent == 0 else (a_seen > 0)
            assert sum(count for label, count in labels.items() if label >= 50) > 0
            totals.update(labels)
        result["training_exposure"][str(percent)] = {
            "local_training_calls": len(calls),
            "optimizer_steps": sum(row["optimizer_steps"] for row in calls),
            "a_samples_processed": sum(count for label, count in totals.items() if label < 50),
            "b_samples_processed": sum(count for label, count in totals.items() if label >= 50),
            "a_classes_seen": sum(label < 50 for label in totals),
            "b_classes_seen": sum(label >= 50 for label in totals),
        }
    result["checks"]["replay_samples_actually_used_in_every_local_training"] = True
    if config.get("train_samples_per_group") is not None:
        expected_steps = math.ceil(config["train_samples_per_group"] / config["batch_size"]) * config["local_epochs"]
        assert all(row["optimizer_steps"] == expected_steps and row["fc_weight_delta_l2"] > 0 for row in audit["training"])
        result["checks"]["real_updates_per_local_training"] = expected_steps
    else:
        sizes = {group: {row["context"]["client"]: row["samples"] for row in audit["datasets"]
                         if row["group"] == group and "client" in row["context"]} for group in ("A", "B")}
        assert config.get("controlled_replay"), "La verificación del presupuesto completo requiere --controlled-replay"
        # Fase 2: mismas actualizaciones que la variante sin replay. Fase 1 (si se
        # entrenó en esta ejecución): una pasada completa por los datos A del cliente.
        assert all(row["optimizer_steps"] == math.ceil(
                       sizes["B" if row["context"].get("phase") == 2 else "A"][row["context"]["client"]] / config["batch_size"]
                   ) * config["local_epochs"] and row["fc_weight_delta_l2"] > 0 for row in audit["training"])
        result["checks"]["equal_updates_for_full_data_per_client_and_round"] = True
    if config.get("validation"):
        assert all(len(group["label_counts"]) == 50 and len(set(group["label_counts"].values())) == 1
                   for group in result["test_groups"].values())
        result["checks"]["minimum_learning_and_balanced_test"] = (
            base >= config["min_base_accuracy_percent"] and
            all(result["evaluations"][f"phase2_{pct}"]["B"]["accuracy_percent"] >= config["min_new_task_accuracy_percent"] for pct in buffers))

    # Verificar la separación de imágenes y la cobertura del conjunto reducido.
    fds = FederatedDataset(dataset="uoft-cs/cifar100", revision=config["dataset_revision"],
                           partitioners={"train": config["num_clients"]}, seed=config["seed"])
    train_groups = {}
    result["train_groups"] = {}
    test_hashes = image_hashes(test_groups["A"]) | image_hashes(test_groups["B"])
    selected_train_hashes = set()
    selected_train_records = Counter()
    hashes_by_train_group = {}
    result["data_overlap"] = {"train_test_shared_hashes_by_client_group": {}}
    for client in range(config["num_clients"]):
        partition = fds.load_partition(client, "train")
        for name, lower, upper in (("A", 0, 50), ("B", 50, 100)):
            group = partition.filter(lambda row: lower <= row["fine_label"] < upper)
            if config.get("train_samples_per_group") is None:
                pass
            elif config.get("sampling") == "balanced":
                group = balanced_subset(group, config["train_samples_per_group"], config["seed"])
            else:
                group = group.shuffle(seed=config["seed"]).select(range(config["train_samples_per_group"]))
            counts = Counter(group["fine_label"])
            key = f"client_{client}_{name}"
            train_groups[key] = group
            result["train_groups"][key] = {"samples": len(group), "classes_present": len(counts),
                                           "label_counts": dict(counts)}
            records = image_records(group)
            hashes = {digest for digest, label in records}
            selected_train_records.update(records)
            hashes_by_train_group[key] = hashes
            selected_train_hashes.update(hashes)
            result["data_overlap"]["train_test_shared_hashes_by_client_group"][key] = sorted(hashes & test_hashes)
    shared = selected_train_hashes & test_hashes
    assert shared <= (official_train_hashes & test_hashes), "Solapamiento ajeno a los splits oficiales"
    result["data_overlap"]["unique_train_test_shared_hashes"] = len(shared)
    result["data_overlap"]["client_0_a_and_client_1_a_shared_hashes"] = sorted(
        hashes_by_train_group["client_0_A"] & hashes_by_train_group["client_1_A"])
    result["checks"]["train_test_shared_images_belong_to_official_source"] = True
    assert all(count <= official_train_records[record] for record, count in selected_train_records.items())
    result["checks"]["training_partitions_preserve_official_image_label_counts"] = True

    if not config["quick"] and not config.get("validation"):
        total_a = sum(len(train_groups[f"client_{client}_A"]) for client in range(config["num_clients"]))
        assert total_a == 25000, "La fase A completa debe contener 25.000 imágenes"
        assert selected_train_hashes == official_train_hashes
        assert selected_train_records == official_train_records
        assert shared == (official_train_hashes & test_hashes)
        result["checks"]["full_training_image_coverage_matches_official_source"] = True
        for percent in buffers:
            record = audit["replay_memory"][str(percent)]
            assert record["a_pool_samples"] == total_a
            expected_memory = sum(len(train_groups[f"client_{client}_A"]) * percent // 100 for client in range(config["num_clients"]))
            assert record["buffer_samples"] == expected_memory
            assert abs(record["actual_fraction"] - expected_memory / total_a) < 1e-12
        result["checks"]["replay_fraction_matches_full_learned_a_pool"] = True
        (args.audit_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("COMPARACIÓN COMPLETA VERIFICADA: métricas, pesos iniciales, presupuesto y memoria real coinciden.", flush=True)
        return

    if config.get("validation"):
        result["validation"] = json.loads((args.audit_dir / "validation.json").read_text())
        result["training_review"] = {}
        train_b_images, train_b_labels = zip(*(tensors(train_groups[f"client_{client}_B"], normalize) for client in range(config["num_clients"])))
        all_b_images, all_b_labels = torch.cat(train_b_images), torch.cat(train_b_labels)
        for percent in buffers:
            model = new_model()
            model.load_state_dict(torch.load(args.audit_dir / f"audit_phase2_{percent}.pt", map_location="cpu", weights_only=True))
            b_metrics = evaluate(model, all_b_images, all_b_labels)
            replay_metrics = None
            if percent:
                replay_xy = []
                for client in range(config["num_clients"]):
                    group = train_groups[f"client_{client}_A"]
                    indices = balanced_order(group["fine_label"], config["seed"] + client)[:len(group) * percent // 100]
                    replay_xy.append(tensors(group.select(indices), normalize))
                replay_metrics = evaluate(model, torch.cat([xy[0] for xy in replay_xy]), torch.cat([xy[1] for xy in replay_xy]))
            review_config = {**config, "min_training_accuracy_percent": config.get("min_training_accuracy_percent", 50.0)}
            problem = learning_problem(result["evaluations"][f"phase2_{percent}"]["B"]["accuracy_percent"],
                                       b_metrics["accuracy_percent"], replay_metrics["accuracy_percent"] if replay_metrics else None,
                                       review_config)
            result["training_review"][str(percent)] = {"training_b": b_metrics, "replay_training": replay_metrics,
                                                       "learning_problem": problem,
                                                       "training_review_threshold_percent": review_config["min_training_accuracy_percent"]}
        result["checks"]["training_learning_controls_passed"] = all(
            row["learning_problem"] is None for row in result["training_review"].values())
        (args.audit_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not result["checks"]["training_learning_controls_passed"] or not result["validation"]["comparison_valid"]:
            print("MÉTRICAS VERIFICADAS, COMPARACIÓN INCONCLUYENTE: fallan los controles de aprendizaje en entrenamiento.", flush=True)
            raise SystemExit(2)
        print("PILOTO VERIFICADO: métricas recalculadas, clases cubiertas, aprendizaje mínimo y presupuesto de actualizaciones iguales.", flush=True)
        return

    # Control positivo: el modelo debe poder memorizar un lote REAL.
    # No se confunde esta precisión de entrenamiento con la de test.
    torch.manual_seed(config["seed"])
    model = new_model()
    images, labels = tensors(train_groups["client_0_A"].select(range(32)), normalize)
    before = evaluate(model, images, labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    losses = []
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            print(f"Control de memorización, paso {step + 1}: loss={loss.item():.6f}", flush=True)
    after = evaluate(model, images, labels)
    result["learning_control"] = {"description": "50 actualizaciones sobre el mismo lote real de 32 imágenes",
                                  "batch_losses": losses, "batch_before": before, "batch_after": after,
                                  "held_out_a_after": evaluate(model, *tensors_by_group["A"])}
    (args.audit_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert losses[-1] < losses[0] * 0.2, "El control no reduce suficientemente la pérdida"
    assert after["accuracy_percent"] >= 90, "El modelo no memoriza el lote del control"
    print("AUDITORÍA SUPERADA: CSV verificado de forma independiente, actualizaciones reales y control de aprendizaje correcto.", flush=True)


if __name__ == "__main__":
    main()
