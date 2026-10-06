import os
import argparse
import csv
import json
import hashlib
from collections import Counter
from pathlib import Path
import math
import gc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from flwr_datasets import FederatedDataset
import uuid
from datetime import datetime
from reproducibility import seed_everything
from cifar100_sampling import balanced_order, balanced_subset
from cifar100_validation import learning_problem, validate_training_contract

# ------------------------------------------------------------------
# 1. Configuration Setup
# ------------------------------------------------------------------
CONFIG = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 10,
    "num_rounds": 5,
    "lr": 0.001,
    "seed": 42,
    "dataset_revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
}

parser = argparse.ArgumentParser(description="Comparativa CIFAR-100 con todos los buffers")
profiles = parser.add_mutually_exclusive_group()
profiles.add_argument("--quick", action="store_true", help="Prueba funcional reducida con datos reales; no sirve para comparar precisión")
profiles.add_argument("--validation", action="store_true", help="Prueba piloto equilibrada con controles de aprendizaje, muestreo y presupuesto de entrenamiento")
parser.add_argument("--output-dir", help="Directorio para nuevos resultados, sin sobrescribir los anteriores")
parser.add_argument("--audit", action="store_true", help="Guarda diagnósticos y pesos para comprobar las métricas de forma independiente")
parser.add_argument("--phase1-checkpoint", type=Path, help="Reutiliza unos pesos de fase A, verificando su precisión")
parser.add_argument("--buffers", nargs="+", type=int, choices=[0, 5, 10, 15, 20, 25], default=[0, 5, 10, 15, 20, 25], help="Porcentajes a comparar")
parser.add_argument("--rounds", type=int, help="Sobrescribe el número de rondas por fase")
parser.add_argument("--local-epochs", type=int, help="Sobrescribe las épocas locales")
parser.add_argument("--controlled-replay", action="store_true", help="Misma semilla, buffers anidados y mismas actualizaciones con los datos completos")
parser.add_argument("--seed", type=int, help="Semilla del experimento (por defecto 42). Repite con varias para medir la variabilidad")
parser.add_argument("--checkpoint-manifest", type=Path, help="Configuración del conjunto A aprendido por el checkpoint")
args = parser.parse_args()
if 0 not in args.buffers or len(args.buffers) != len(set(args.buffers)):
    parser.error("--buffers debe incluir 0 y no repetir porcentajes")
args.buffers.sort()
if args.phase1_checkpoint:
    args.phase1_checkpoint = args.phase1_checkpoint.resolve()
    if not args.phase1_checkpoint.is_file():
        parser.error("No existe el checkpoint de fase A")
if args.quick:
    CONFIG.update(num_clients=2, local_epochs=1, num_rounds=1)
    CONFIG.update(train_samples_per_group=100, test_samples_per_group=100)
    print("QUICK TEST: muestras reducidas; las precisiones no son resultados del experimento completo.")
if args.validation:
    CONFIG.update(num_clients=2, local_epochs=3, num_rounds=3)
    CONFIG.update(train_samples_per_group=500, test_samples_per_group=1000,
                  sampling="balanced", min_base_accuracy_percent=10.0,
                  min_new_task_accuracy_percent=8.0, min_training_accuracy_percent=50.0,
                  equal_optimizer_steps=True)
    print("VALIDATION: prueba piloto; aprendizaje mínimo, clases cubiertas y entrenamiento controlado.")
for argument, key in ((args.rounds, "num_rounds"), (args.local_epochs, "local_epochs")):
    if argument is not None:
        if argument < 1:
            parser.error("Las rondas y épocas deben ser positivas")
        CONFIG[key] = argument
if args.seed is not None:
    CONFIG["seed"] = args.seed
controlled_replay = args.validation or args.controlled_replay
checkpoint_contract = None
if args.phase1_checkpoint:
    candidates = [args.checkpoint_manifest] if args.checkpoint_manifest else [
        args.phase1_checkpoint.with_suffix(".config.json"), args.phase1_checkpoint.parent / "config.json"]
    manifest_path = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
    if manifest_path is None:
        parser.error("El checkpoint necesita su configuración de entrenamiento A (--checkpoint-manifest); no se puede suponer el tamaño de su replay")
    checkpoint_contract = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    try:
        validate_training_contract(CONFIG, checkpoint_contract)
    except ValueError as error:
        parser.error(str(error) + ". Usa los mismos datos A y clientes; no recortes la memoria de un checkpoint completo.")
    checkpoint_sha256 = hashlib.sha256(args.phase1_checkpoint.read_bytes()).hexdigest()
    if checkpoint_contract.get("checkpoint_sha256") and checkpoint_contract["checkpoint_sha256"] != checkpoint_sha256:
        parser.error("El manifiesto no corresponde al SHA-256 de este checkpoint")
if not args.output_dir:
    # Nunca escribir en la raíz: así no se sobrescriben los gráficos históricos.
    args.output_dir = os.path.join("resultados", f"cifar100_three_buffer_seed{CONFIG['seed']}_{datetime.now():%Y-%m-%d_%H%M%S}")
os.makedirs(args.output_dir, exist_ok=True)
os.chdir(args.output_dir)
print(f"[INFO] Resultados en: {os.getcwd()}")
with open("config.json", "w", encoding="utf-8") as config_file:
    json.dump({**CONFIG, "quick": args.quick, "validation": args.validation,
               "audit": args.audit, "buffers": args.buffers, "controlled_replay": controlled_replay,
               "phase1_training_contract": checkpoint_contract, "torch_version": torch.__version__,
               "cpu_threads": torch.get_num_threads(),
               "phase1_checkpoint": str(args.phase1_checkpoint) if args.phase1_checkpoint else None,
               "phase1_checkpoint_sha256": hashlib.sha256(args.phase1_checkpoint.read_bytes()).hexdigest() if args.phase1_checkpoint else None}, config_file, indent=2)

seed_everything(CONFIG["seed"])
print(f"Device in use: {CONFIG['device']}")
audit = {"datasets": [], "training": [], "evaluation": [], "phase2_initial_weights": {}}
audit_context = {}
round_history = []
validation_status = {"comparison_valid": False, "profile": "quick" if args.quick else "validation" if args.validation else "full",
                     "reason": "Prueba funcional sin validación de aprendizaje" if args.quick else "Pendiente de completar"}

def save_validation_status():
    with open("validation.json", "w", encoding="utf-8") as status_file:
        json.dump(validation_status, status_file, indent=2)

def save_audit():
    if args.audit:
        with open("audit.json", "w", encoding="utf-8") as audit_file:
            json.dump(audit, audit_file, indent=2)

save_validation_status()

def weights_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

# ------------------------------------------------------------------
# 2. Data Preparation & Filtering
# ------------------------------------------------------------------
fds = FederatedDataset(
    dataset="uoft-cs/cifar100",
    revision=CONFIG["dataset_revision"],
    partitioners={"train": CONFIG["num_clients"]},
    seed=CONFIG["seed"],
)

group_a_classes = list(range(0, 50))
group_b_classes = list(range(50, 100))

def filter_by_classes(partition, class_labels, sample_limit=None):
    allowed_classes = set(class_labels)
    label_col = "fine_label" if "fine_label" in partition.column_names else "label"
    filtered = partition.filter(lambda example: example[label_col] in allowed_classes)
    if sample_limit is not None and len(filtered) > sample_limit:
        if CONFIG.get("sampling") == "balanced":
            filtered = balanced_subset(filtered, sample_limit, CONFIG["seed"])
        else:
            filtered = filtered.shuffle(seed=CONFIG["seed"]).select(range(sample_limit))
    if args.audit:
        counts = Counter(filtered[label_col])
        audit["datasets"].append({
            "context": dict(audit_context), "group": "A" if min(class_labels) == 0 else "B",
            "samples": len(filtered), "classes_present": len(counts), "label_counts": dict(counts),
        })
    return filtered

def transform_batch(batch):
    images = [
        torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        for img in batch["img"]
    ]
    labels = torch.tensor(batch["fine_label"] if "fine_label" in batch else batch["label"], dtype=torch.long)
    return {"img": images, "label": labels}

def collate_fn(batch):
    imgs = torch.stack([x["img"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return imgs, labels

# Prepare Test Loaders for Group A and Group B
test_dataset = fds.load_split("test")

test_group_a = filter_by_classes(test_dataset, group_a_classes, CONFIG.get("test_samples_per_group"))
test_group_b = filter_by_classes(test_dataset, group_b_classes, CONFIG.get("test_samples_per_group"))

test_loader_a = DataLoader(
    test_group_a.with_transform(transform_batch), 
    batch_size=CONFIG["batch_size"], 
    collate_fn=collate_fn
)

test_loader_b = DataLoader(
    test_group_b.with_transform(transform_batch), 
    batch_size=CONFIG["batch_size"], 
    collate_fn=collate_fn
)

# ------------------------------------------------------------------
# 3. Helper Functions
# ------------------------------------------------------------------
def get_model():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def train_local(model, train_loader, epochs, lr, device, max_steps=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if args.audit:
        before = model.fc.weight.detach().clone()
        losses = []
        labels_seen = Counter()
    
    steps = 0
    for _ in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            steps += 1
            if args.audit:
                if not torch.isfinite(loss):
                    raise RuntimeError("La pérdida de entrenamiento no es finita")
                losses.append(loss.item())
                labels_seen.update(labels.detach().cpu().tolist())
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break

    if args.audit:
        audit["training"].append({
            "context": dict(audit_context), "optimizer_steps": len(losses),
            "samples_processed": sum(labels_seen.values()), "classes_seen": len(labels_seen),
            "label_counts": dict(labels_seen), "batch_losses": losses,
            "fc_weight_delta_l2": (model.fc.weight.detach() - before).norm().item(),
        })
            
    return model.state_dict()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    if args.audit:
        prediction_counts = Counter()
        target_counts = Counter()
        loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if args.audit:
                prediction_counts.update(predicted.detach().cpu().tolist())
                target_counts.update(labels.detach().cpu().tolist())
                loss_sum += nn.functional.cross_entropy(outputs, labels, reduction="sum").item()
    if args.audit:
        if total == 0:
            raise RuntimeError("La evaluación no contiene muestras")
        audit["evaluation"].append({
            "context": dict(audit_context), "correct": correct, "total": total,
            "accuracy_percent": 100 * correct / total, "mean_cross_entropy": loss_sum / total,
            "prediction_counts": dict(prediction_counts), "target_counts": dict(target_counts),
        })
    return correct / total if total > 0 else 0.0

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_table(title, results_list):
    print("\n" + "="*95)
    print(f"                    {title}")
    print("="*95)
    print(f"{'Method':<20} | {'Group A (Before)':<18} | {'Group A (After)':<18} | {'Group B (After)':<18} | {'Accuracy Loss':<15}")
    print("-" * 95)
    for res in results_list:
        print(f"{res['fraction']:<20} | {res['step3_acc_a']:>17.2f}% | {res['step4_acc_a']:>17.2f}% | {res['step4_acc_b']:>17.2f}% | {res['loss_a']:>12.2f} pp")
    print("="*95)

def plot_and_save_chart(results_list, filename, title_text):
    if args.quick:
        print("[INFO] Prueba funcional: se omite el gráfico comparativo porque no valida aprendizaje.")
        return
    methods = [res['fraction'] for res in results_list]
    accuracy_losses = [res['loss_a'] for res in results_list]
    colors = ['#e74c3c', '#3498db', '#f1c40f', '#2ecc71', '#e67e22', '#9b59b6'][:len(results_list)]

    plt.figure(figsize=(9, 5.5), dpi=300)
    bars = plt.bar(methods, accuracy_losses, color=colors, width=0.45, edgecolor='black', linewidth=1)

    plt.title(title_text, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Method', fontsize=12, labelpad=10)
    plt.ylabel('Accuracy Loss (percentage points)', fontsize=12, labelpad=10)
    plt.ylim(min(0, min(accuracy_losses) - 4), max(0, max(accuracy_losses)) + 5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        # Las pérdidas negativas (A mejora) se etiquetan por debajo de la barra.
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (0.7 if yval >= 0 else -0.7), f'{yval:.2f} pp',
                 ha='center', va='bottom' if yval >= 0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[INFO] Chart successfully saved as '{filename}'")
    plt.close()

# ------------------------------------------------------------------
# 4. PHASE 1: Single Base Training on Group A
# ------------------------------------------------------------------
print("\n========================================================")
print("  PHASE 1: TRAINING ON GROUP A (ONCE FOR ALL EXPERIMENTS)")
print("========================================================")

global_model = get_model().to(CONFIG["device"])
if args.phase1_checkpoint:
    global_model.load_state_dict(torch.load(args.phase1_checkpoint, map_location=CONFIG["device"], weights_only=True))
    print(f"[INFO] Fase A reutilizada de: {args.phase1_checkpoint}")
client_group_a_datasets = {}
client_group_b_datasets = {}

for client_id in range(CONFIG["num_clients"]):
    audit_context.update(phase=1, client=client_id)
    client_partition = fds.load_partition(partition_id=client_id, split="train")
    client_group_a_datasets[client_id] = filter_by_classes(client_partition, group_a_classes, CONFIG.get("train_samples_per_group"))
    if controlled_replay:
        client_group_b_datasets[client_id] = filter_by_classes(client_partition, group_b_classes, CONFIG.get("train_samples_per_group"))

data_manifest = {"clients": [{"client": client, "a_samples": len(dataset),
                             "b_samples": len(client_group_b_datasets[client]) if client in client_group_b_datasets else None}
                            for client, dataset in client_group_a_datasets.items()],
                 "total_a_samples": sum(len(dataset) for dataset in client_group_a_datasets.values())}
with open("data_manifest.json", "w", encoding="utf-8") as manifest_file:
    json.dump(data_manifest, manifest_file, indent=2)
print(f"[DATA] El buffer se calcula sobre {data_manifest['total_a_samples']} imágenes A en total.")

for r in range(0 if args.phase1_checkpoint else CONFIG["num_rounds"]):
    local_weights = []
    
    for client_id in range(CONFIG["num_clients"]):
        audit_context.update(phase=1, round=r + 1, client=client_id)
        client_group_a = client_group_a_datasets[client_id]
        
        train_loader = DataLoader(
            client_group_a.with_transform(transform_batch), 
            batch_size=CONFIG["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        local_model = get_model().to(CONFIG["device"])
        local_model.load_state_dict(global_model.state_dict())
        
        w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"])
        local_weights.append(w)
    
    avg_weights = {}
    for key in local_weights[0].keys():
        target_dtype = local_weights[0][key].dtype
        stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
        avg_tensor = stacked_weights.mean(dim=0)
        avg_weights[key] = avg_tensor.to(target_dtype)

    global_model.load_state_dict(avg_weights)
    acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
    print(f"Phase 1 - Round {r+1}/{CONFIG['num_rounds']} -> Group A Accuracy: {acc_a * 100:.2f}%")

base_group_a_weights = copy.deepcopy(global_model.state_dict())
base_acc_a = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100
print(f"[INFO] Precisión A antes de comparar replay: {base_acc_a:.2f}%")
validation_status["base_accuracy_a_percent"] = base_acc_a
if args.validation and base_acc_a < CONFIG["min_base_accuracy_percent"]:
    validation_status["reason"] = "Aprendizaje insuficiente en A; se cancela la comparación de replay"
    save_validation_status()
    save_audit()
    raise RuntimeError(validation_status["reason"])
if args.audit:
    torch.save(base_group_a_weights, "audit_phase1.pt")
    audit["phase1_weights_sha256"] = weights_digest(global_model)
    save_audit()

# ------------------------------------------------------------------
# 5. PHASE 2: Runner Function for Replay Buffer Experiments
# ------------------------------------------------------------------
def run_phase2_experiment(replay_fraction):
    if controlled_replay:
        seed_everything(CONFIG["seed"])
    fraction_label = "No Replay" if replay_fraction == 0.0 else f"{int(replay_fraction * 100)}% Replay"
    print(f"\n========================================================")
    print(f"  RUNNING PHASE 2: Replay Buffer = {fraction_label}")
    print(f"========================================================")

    global_model = get_model().to(CONFIG["device"])
    global_model.load_state_dict(copy.deepcopy(base_group_a_weights))
    if args.audit:
        audit["phase2_initial_weights"][str(replay_fraction)] = weights_digest(global_model)

    client_replay_buffers = {}
    for client_id in range(CONFIG["num_clients"]):
        client_group_a = client_group_a_datasets[client_id]
        buffer_size = int(len(client_group_a) * replay_fraction)
        if buffer_size > 0:
            if controlled_replay:
                label_col = "fine_label" if "fine_label" in client_group_a.column_names else "label"
                shuffled_indices = balanced_order(client_group_a[label_col], CONFIG["seed"] + client_id)[:buffer_size]
            else:
                shuffled_indices = np.random.choice(len(client_group_a), size=buffer_size, replace=False)
            client_replay_buffers[client_id] = client_group_a.select(shuffled_indices)
        else:
            client_replay_buffers[client_id] = None

    replay_total = sum(len(dataset) for dataset in client_replay_buffers.values() if dataset is not None)
    audit.setdefault("replay_memory", {})[str(int(replay_fraction * 100))] = {
        "a_pool_samples": data_manifest["total_a_samples"], "buffer_samples": replay_total,
        "actual_fraction": replay_total / data_manifest["total_a_samples"],
        "per_client": [{"client": client, "a_samples": len(client_group_a_datasets[client]),
                        "buffer_samples": len(dataset) if dataset is not None else 0}
                       for client, dataset in client_replay_buffers.items()]}
    print(f"[REPLAY] Memoria real: {replay_total}/{data_manifest['total_a_samples']} imágenes A.")

    for r in range(CONFIG["num_rounds"]):
        local_weights = []
        
        for client_id in range(CONFIG["num_clients"]):
            audit_context.clear()
            audit_context.update(phase=2, replay_fraction=replay_fraction, round=r + 1, client=client_id)
            client_partition = fds.load_partition(partition_id=client_id, split="train")
            client_group_b = client_group_b_datasets[client_id] if controlled_replay else filter_by_classes(client_partition, group_b_classes, CONFIG.get("train_samples_per_group"))
            
            datasets_to_combine = [client_group_b.with_transform(transform_batch)]
            
            if client_replay_buffers[client_id] is not None:
                datasets_to_combine.append(client_replay_buffers[client_id].with_transform(transform_batch))
            
            combined_dataset = ConcatDataset(datasets_to_combine)
            
            train_loader = DataLoader(
                combined_dataset, 
                batch_size=CONFIG["batch_size"], 
                shuffle=True, 
                collate_fn=collate_fn
            )
            
            local_model = get_model().to(CONFIG["device"])
            local_model.load_state_dict(global_model.state_dict())
            
            max_steps = math.ceil(len(client_group_b) / CONFIG["batch_size"]) * CONFIG["local_epochs"] if controlled_replay else None
            print(f"[CLIENT] Buffer {int(replay_fraction * 100)}%, round {r + 1}, client {client_id + 1}/{CONFIG['num_clients']}: {max_steps or 'epoch budget'} updates", flush=True)
            w = train_local(local_model, train_loader, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["device"], max_steps=max_steps)
            local_weights.append(w)
        
        avg_weights = {}
        for key in local_weights[0].keys():
            target_dtype = local_weights[0][key].dtype
            stacked_weights = torch.stack([w[key].to(torch.float32) for w in local_weights], dim=0)
            avg_tensor = stacked_weights.mean(dim=0)
            avg_weights[key] = avg_tensor.to(target_dtype)

        global_model.load_state_dict(avg_weights)
        
        acc_a = evaluate(global_model, test_loader_a, CONFIG["device"])
        acc_b = evaluate(global_model, test_loader_b, CONFIG["device"])
        
        print(f"Phase 2 - Round {r+1}/{CONFIG['num_rounds']} -> Group A Accuracy: {acc_a * 100:.2f}% | Group B Accuracy: {acc_b * 100:.2f}%")
        round_history.append({"replay_percent": int(replay_fraction * 100), "round": r + 1,
                              "accuracy_a_percent": acc_a * 100, "accuracy_b_percent": acc_b * 100})
        with open("history.csv", "w", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(round_history[0]))
            writer.writeheader()
            writer.writerows(round_history)

    acc_a_after_step4 = evaluate(global_model, test_loader_a, CONFIG["device"]) * 100
    acc_b_after_step4 = evaluate(global_model, test_loader_b, CONFIG["device"]) * 100
    if args.audit:
        torch.save(global_model.state_dict(), f"audit_phase2_{int(replay_fraction * 100)}.pt")

    training_metrics = {}
    if args.validation:
        training_b = ConcatDataset([dataset.with_transform(transform_batch) for dataset in client_group_b_datasets.values()])
        loader_b = DataLoader(training_b, batch_size=CONFIG["batch_size"], collate_fn=collate_fn)
        training_metrics["train_acc_b"] = evaluate(global_model, loader_b, CONFIG["device"]) * 100
        replay_sets = [dataset.with_transform(transform_batch) for dataset in client_replay_buffers.values() if dataset is not None]
        training_metrics["train_acc_replay"] = None
        if replay_sets:
            loader_replay = DataLoader(ConcatDataset(replay_sets), batch_size=CONFIG["batch_size"], collate_fn=collate_fn)
            training_metrics["train_acc_replay"] = evaluate(global_model, loader_replay, CONFIG["device"]) * 100
        print(f"[CONTROL] B en entrenamiento: {training_metrics['train_acc_b']:.2f}%; replay: {training_metrics['train_acc_replay']}")

    clear_memory()

    return {
        "fraction": fraction_label,
        "step3_acc_a": base_acc_a,
        "step4_acc_a": acc_a_after_step4,
        "step4_acc_b": acc_b_after_step4,
        "loss_a": base_acc_a - acc_a_after_step4,
        **training_metrics,
    }

# ------------------------------------------------------------------
# 6. STAGE 1 & STAGE 2 EXPERIMENTS (0%, 5%, 10%, 15%, 20%, 25%)
# ------------------------------------------------------------------
results_stage_2 = []
for percent in args.buffers:
    result = run_phase2_experiment(percent / 100)
    results_stage_2.append(result)
    save_audit()
    if args.validation:
        with open("diagnostic_results.csv", "w", newline="", encoding="utf-8") as diagnostic_file:
            writer = csv.DictWriter(diagnostic_file, fieldnames=list(results_stage_2[0]))
            writer.writeheader()
            writer.writerows(results_stage_2)
        problem = learning_problem(result["step4_acc_b"], result["train_acc_b"], result["train_acc_replay"], CONFIG)
        if problem:
            validation_status["reason"] = f"{problem} con buffer {percent}%; comparación no validada"
            save_validation_status()
            raise RuntimeError(validation_status["reason"])
validation_status.update(comparison_valid=args.validation,
                         reason="Controles mínimos del piloto superados; no sustituye varias semillas ni el experimento completo" if args.validation else "Prueba funcional sin validación de aprendizaje" if args.quick else "Ejecución completa; revisar aprendizaje y replicar con varias semillas")
save_validation_status()
results_stage_1 = [result for result, percent in zip(results_stage_2, args.buffers) if percent <= 20]
print_table("COMPARISON TABLE (SELECTED BUFFERS)", results_stage_2)
plot_and_save_chart(results_stage_1, "group_a_accuracy_loss_up_to_20.png", "Group A Accuracy Loss (Selected Buffers up to 20%)")
with open("results.csv", "w", newline="", encoding="utf-8") as results_file:
    writer = csv.DictWriter(results_file, fieldnames=list(results_stage_2[0]))
    writer.writeheader()
    writer.writerows(results_stage_2)
print_table("FINAL COMPARISON TABLE (SELECTED REPLAY BUFFERS)", results_stage_2)
plot_and_save_chart(results_stage_2, "group_a_accuracy_loss_all_buffers.png", "Group A Accuracy Loss Across Selected Buffers")
if args.audit:
    save_audit()
    print("[AUDIT] Diagnósticos guardados en audit.json y pesos en audit_phase*.pt")
