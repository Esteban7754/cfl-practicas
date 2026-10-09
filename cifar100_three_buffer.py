"""CIFAR-100 federado continuo: fase A (clases 0-49) y fase B (50-99) con replay.

Perfiles:
  (por defecto)  datos completos, 10 clientes; replay controlado (mismos pesos
                 iniciales, buffers anidados y mismo número de actualizaciones).
  --quick        prueba funcional mínima; no sirve para comparar.
  --validation   piloto reducido con controles de aprendizaje.

Variantes de replay (combinables):
  --replay-mix concat     diseño original: B y buffer barajados juntos.
  --replay-mix balanced   cada lote lleva una proporción fija de muestras del
                          buffer (--replay-batch-fraction, 0.5 por defecto).
  --distill-weight L      añade destilación (LwF) sobre los logits de A.

Referencias y métricas añadidas a cada ejecución:
  Weight Aligning (WA)    corrección posterior del sesgo de la capa final hacia B,
                          evaluada sobre el mismo modelo (columnas wa_*).
  avg_acc_tasks, bwt_a    precisión media por tarea y backward transfer.
  --joint                 cota superior: A y B a la vez desde cero con el mismo
                          presupuesto (joint_results.csv y columnas *_vs_joint).

Preprocesado: normalización con las medias de CIFAR-100 y aumento de datos
(recorte con relleno y volteo) activados por defecto; --no-normalize y
--no-augment reproducen el protocolo antiguo (necesario para reutilizar
global_model_phase1.pt, entrenado sin ellos).
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

import fl_core
from cifar100_sampling import balanced_order, balanced_subset
from cifar100_validation import learning_problem, validate_training_contract
from experiment_common import plot_loss
from reproducibility import environment_info, seed_everything

GROUP_A = list(range(0, 50))
GROUP_B = list(range(50, 100))
DATASET = "uoft-cs/cifar100"


def use_preloaded_data():
    """Con CIFAR-100 dentro de la imagen Docker, no depende de Internet (salvo que se pida explícitamente)."""
    if os.environ.get("CFL_DATA_PRELOADED") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def default_config():
    return {
        "num_clients": 10,
        "batch_size": 32,
        "local_epochs": 10,
        "num_rounds": 5,
        "lr": 0.001,
        "seed": 42,
        "dataset_revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
        "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    }


def parse_percent(value):
    percent = int(value)
    if not 0 <= percent < 100:
        raise argparse.ArgumentTypeError("los porcentajes de buffer deben estar entre 0 y 99")
    return percent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--quick", action="store_true", help="Prueba funcional reducida; no sirve para comparar precisión")
    profiles.add_argument("--validation", action="store_true", help="Piloto equilibrado con controles de aprendizaje")
    parser.add_argument("--output-dir", help="Directorio para nuevos resultados, sin sobrescribir los anteriores")
    parser.add_argument("--audit", action="store_true", help="Guarda diagnósticos y pesos para verificar_cifar100.py")
    parser.add_argument("--phase1-checkpoint", type=Path, help="Reutiliza unos pesos de fase A (requiere su manifiesto)")
    parser.add_argument("--checkpoint-manifest", type=Path, help="Configuración del conjunto A aprendido por el checkpoint")
    parser.add_argument("--buffers", nargs="+", type=parse_percent, default=[0, 5, 10, 15, 20, 25],
                        help="Porcentajes de A conservados por cliente; 0 (sin replay) es obligatorio")
    parser.add_argument("--rounds", type=int, help="Sobrescribe el número de rondas por fase")
    parser.add_argument("--local-epochs", type=int, help="Sobrescribe las épocas locales")
    parser.add_argument("--seed", type=int, help="Semilla del experimento (por defecto 42)")
    parser.add_argument("--controlled-replay", action=argparse.BooleanOptionalAction, default=True,
                        help="Mismas actualizaciones, semilla reiniciada y buffers anidados (por defecto). "
                             "--no-controlled-replay recupera el diseño antiguo, en el que más buffer = más pasos")
    parser.add_argument("--replay-mix", choices=["concat", "balanced"], default="concat",
                        help="Cómo se mezclan B y el buffer en cada lote")
    parser.add_argument("--replay-batch-fraction", type=float, default=0.5,
                        help="Con --replay-mix balanced: fracción de cada lote que sale del buffer")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="Peso de la destilación sobre las clases A respecto al modelo de la fase A (0 = sin destilación)")
    parser.add_argument("--joint", action="store_true",
                        help="Añade la cota superior: entrenamiento conjunto A+B desde cero con el mismo presupuesto")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Normaliza con media/desviación de CIFAR-100")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Recorte aleatorio con relleno y volteo horizontal durante el entrenamiento")
    return parser


def resolve_config(args, parser):
    config = default_config()
    if 0 not in args.buffers or len(args.buffers) != len(set(args.buffers)):
        parser.error("--buffers debe incluir 0 y no repetir porcentajes")
    args.buffers.sort()
    if args.quick:
        config.update(num_clients=2, local_epochs=1, num_rounds=1, train_samples_per_group=100, test_samples_per_group=100)
    if args.validation:
        config.update(num_clients=2, local_epochs=3, num_rounds=3, train_samples_per_group=500,
                      test_samples_per_group=1000, sampling="balanced", min_base_accuracy_percent=10.0,
                      min_new_task_accuracy_percent=8.0, min_training_accuracy_percent=50.0,
                      equal_optimizer_steps=True)
    for value, key in ((args.rounds, "num_rounds"), (args.local_epochs, "local_epochs")):
        if value is not None:
            if value < 1:
                parser.error("Las rondas y épocas deben ser positivas")
            config[key] = value
    if args.seed is not None:
        config["seed"] = args.seed
    controlled = args.validation or args.controlled_replay
    if args.replay_mix == "balanced":
        if not controlled:
            parser.error("--replay-mix balanced necesita el presupuesto controlado de actualizaciones")
        if not 0 < args.replay_batch_fraction < 1:
            parser.error("--replay-batch-fraction debe estar entre 0 y 1")
    if args.distill_weight < 0:
        parser.error("--distill-weight no puede ser negativo")
    config.update(normalize=args.normalize, augment=args.augment, controlled_replay=controlled,
                  replay_mix=args.replay_mix,
                  replay_batch_fraction=args.replay_batch_fraction if args.replay_mix == "balanced" else None,
                  distill_weight=args.distill_weight, joint_baseline=args.joint)

    contract = None
    if args.phase1_checkpoint:
        args.phase1_checkpoint = args.phase1_checkpoint.resolve()
        if not args.phase1_checkpoint.is_file():
            parser.error("No existe el checkpoint de fase A")
        candidates = [args.checkpoint_manifest] if args.checkpoint_manifest else [
            args.phase1_checkpoint.with_suffix(".config.json"), args.phase1_checkpoint.parent / "config.json"]
        manifest = next((c for c in candidates if c and c.is_file()), None)
        if manifest is None:
            parser.error("El checkpoint necesita su configuración de entrenamiento A (--checkpoint-manifest)")
        contract = json.loads(manifest.read_text(encoding="utf-8-sig"))
        try:
            validate_training_contract(config, contract)
        except ValueError as error:
            parser.error(f"{error}. Usa los mismos datos, clientes y preprocesado con los que se entrenó el checkpoint "
                         "(global_model_phase1.pt requiere --no-normalize --no-augment).")
        sha = hashlib.sha256(args.phase1_checkpoint.read_bytes()).hexdigest()
        if contract.get("checkpoint_sha256") and contract["checkpoint_sha256"] != sha:
            parser.error("El manifiesto no corresponde al SHA-256 de este checkpoint")
    return config, contract


class Experiment:
    def __init__(self, args, config, contract):
        self.args, self.config, self.contract = args, config, contract
        self.device = config["device"]
        self.audit = {"datasets": [], "training": [], "evaluation": [], "phase2_initial_weights": {}}
        self.context = {}
        self.history = []
        self.status = {"comparison_valid": False,
                       "profile": "quick" if args.quick else "validation" if args.validation else "full",
                       "reason": "Prueba funcional sin validación de aprendizaje" if args.quick else "Pendiente de completar"}
        self.preprocess = fl_core.Preprocess(normalize=config["normalize"])

    # ---------------------------------------------------------- persistencia
    def save_status(self):
        Path("validation.json").write_text(json.dumps(self.status, indent=2), encoding="utf-8")

    def save_audit(self):
        if self.args.audit:
            Path("audit.json").write_text(json.dumps(self.audit, indent=2), encoding="utf-8")

    # ---------------------------------------------------------- datos
    def filter_by_classes(self, partition, classes, limit=None):
        allowed = set(classes)
        label_col = "fine_label" if "fine_label" in partition.column_names else "label"
        filtered = partition.filter(lambda example: example[label_col] in allowed)
        if limit is not None and len(filtered) > limit:
            if self.config.get("sampling") == "balanced":
                filtered = balanced_subset(filtered, limit, self.config["seed"])
            else:
                filtered = filtered.shuffle(seed=self.config["seed"]).select(range(limit))
        if self.args.audit:
            counts = Counter(filtered[label_col])
            self.audit["datasets"].append({
                "context": dict(self.context), "group": "A" if min(classes) == 0 else "B",
                "samples": len(filtered), "classes_present": len(counts), "label_counts": dict(counts)})
        return fl_core.hf_to_tensors(filtered)

    def load_data(self):
        use_preloaded_data()
        from flwr_datasets import FederatedDataset

        cfg = self.config
        fds = FederatedDataset(dataset=DATASET, revision=cfg["dataset_revision"],
                               partitioners={"train": cfg["num_clients"]}, seed=cfg["seed"])
        test = fds.load_split("test")
        self.test_a = self.filter_by_classes(test, GROUP_A, cfg.get("test_samples_per_group"))
        self.test_b = self.filter_by_classes(test, GROUP_B, cfg.get("test_samples_per_group"))
        self.train_a, self.train_b = {}, {}
        for client in range(cfg["num_clients"]):
            self.context.update(phase=1, client=client)
            partition = fds.load_partition(partition_id=client, split="train")
            self.train_a[client] = self.filter_by_classes(partition, GROUP_A, cfg.get("train_samples_per_group"))
            self.train_b[client] = self.filter_by_classes(partition, GROUP_B, cfg.get("train_samples_per_group"))
        manifest = {"clients": [{"client": c, "a_samples": len(self.train_a[c][1]), "b_samples": len(self.train_b[c][1])}
                                for c in self.train_a],
                    "total_a_samples": sum(len(y) for _, y in self.train_a.values()),
                    "preprocess": self.preprocess.describe(), "augment": cfg["augment"]}
        Path("data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.total_a = manifest["total_a_samples"]
        print(f"[DATA] El buffer se calcula sobre {self.total_a} imágenes A en total.")

    # ---------------------------------------------------------- entrenamiento
    def evaluate(self, model, x, y):
        result = fl_core.evaluate(model, x, y, self.device, self.preprocess, group_a_classes=GROUP_A)
        if self.args.audit:
            self.audit["evaluation"].append({"context": dict(self.context), **result})
        return result

    def local_round(self, global_model, batches_for_client, generator, teacher=None, audit=True):
        weights = []
        for client in range(self.config["num_clients"]):
            self.context["client"] = client
            local = fl_core.get_model().to(self.device)
            local.load_state_dict(global_model.state_dict())
            record = {} if self.args.audit and audit else None
            weights.append(fl_core.train_local(
                local, batches_for_client(client), self.config["lr"], self.device, self.preprocess,
                augment=self.config["augment"], generator=generator, teacher=teacher,
                distill_weight=self.config["distill_weight"], old_classes=GROUP_A, record=record))
            if record is not None:
                self.audit["training"].append({"context": dict(self.context), **record})
        global_model.load_state_dict(fl_core.fedavg(weights))

    def phase1(self):
        cfg, args = self.config, self.args
        print("\n=== FASE 1: ENTRENAMIENTO EN EL GRUPO A (una vez para todas las variantes) ===")
        model = fl_core.get_model().to(self.device)
        if args.phase1_checkpoint:
            model.load_state_dict(torch.load(args.phase1_checkpoint, map_location=self.device, weights_only=True))
            print(f"[INFO] Fase A reutilizada de: {args.phase1_checkpoint}")
        else:
            generator = torch.Generator().manual_seed(cfg["seed"])
            for r in range(cfg["num_rounds"]):
                self.context = {"phase": 1, "round": r + 1}

                def batches(client):
                    x, y = self.train_a[client]
                    return fl_core.concat_batches(x, y, None, None, cfg["batch_size"], cfg["local_epochs"], generator=generator)

                self.local_round(model, batches, generator)
                acc = self.evaluate(model, *self.test_a)["accuracy_percent"]
                print(f"Fase 1 - Ronda {r + 1}/{cfg['num_rounds']} -> Precisión A: {acc:.2f}%")
        self.context = {"phase": 1, "final": True}
        self.base_weights = copy.deepcopy(model.state_dict())
        self.base_acc_a = self.evaluate(model, *self.test_a)["accuracy_percent"]
        print(f"[INFO] Precisión A antes de comparar replay: {self.base_acc_a:.2f}%")
        self.status["base_accuracy_a_percent"] = self.base_acc_a
        if args.validation and self.base_acc_a < cfg["min_base_accuracy_percent"]:
            self.status["reason"] = "Aprendizaje insuficiente en A; se cancela la comparación de replay"
            self.save_status()
            self.save_audit()
            raise RuntimeError(self.status["reason"])
        if not args.phase1_checkpoint and not args.quick:
            self.save_phase1_checkpoint(model)
        if args.audit:
            torch.save(self.base_weights, "audit_phase1.pt")
            self.audit["phase1_weights_sha256"] = fl_core.weights_digest(model)
            self.save_audit()

    def save_phase1_checkpoint(self, model):
        """Guarda los pesos A con su manifiesto, para poder reutilizarlos con trazabilidad."""
        torch.save(model.state_dict(), "phase1_checkpoint.pt")
        keys = ("num_clients", "batch_size", "local_epochs", "num_rounds", "lr", "seed", "dataset_revision",
                "train_samples_per_group", "sampling", "normalize", "augment")
        manifest = {key: self.config.get(key) for key in keys}
        manifest.update(group_a_classes=GROUP_A,
                        checkpoint_sha256=hashlib.sha256(Path("phase1_checkpoint.pt").read_bytes()).hexdigest(),
                        provenance=f"Generado por cifar100_three_buffer.py el {datetime.now():%Y-%m-%d %H:%M}",
                        phase1_test_accuracy_percent=self.base_acc_a, torch_version=torch.__version__)
        Path("phase1_checkpoint.config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def build_buffers(self, percent):
        buffers = {}
        for client, (x, y) in self.train_a.items():
            size = len(y) * percent // 100
            if size == 0:
                buffers[client] = None
                continue
            if self.config["controlled_replay"]:
                indices = balanced_order(y.tolist(), self.config["seed"] + client)[:size]
            else:
                rng = torch.Generator().manual_seed(self.config["seed"] * 1000 + client)
                indices = torch.randperm(len(y), generator=rng)[:size].tolist()
            idx = torch.as_tensor(indices)
            buffers[client] = (x[idx], y[idx])
        return buffers

    def phase2(self, percent):
        cfg = self.config
        if cfg["controlled_replay"]:
            seed_everything(cfg["seed"])
        label = "No Replay" if percent == 0 else f"{percent}% Replay"
        print(f"\n=== FASE 2: buffer = {label} ({cfg['replay_mix']}, destilación {cfg['distill_weight']}) ===")
        model = fl_core.get_model().to(self.device)
        model.load_state_dict(copy.deepcopy(self.base_weights))
        if self.args.audit:
            self.audit["phase2_initial_weights"][str(percent / 100)] = fl_core.weights_digest(model)
        teacher = None
        if cfg["distill_weight"] > 0:
            teacher = fl_core.get_model().to(self.device)
            teacher.load_state_dict(self.base_weights)
            for p in teacher.parameters():
                p.requires_grad_(False)

        buffers = self.build_buffers(percent)
        replay_total = sum(len(b[1]) for b in buffers.values() if b is not None)
        self.audit.setdefault("replay_memory", {})[str(percent)] = {
            "a_pool_samples": self.total_a, "buffer_samples": replay_total,
            "actual_fraction": replay_total / self.total_a,
            "per_client": [{"client": c, "a_samples": len(self.train_a[c][1]),
                            "buffer_samples": len(b[1]) if b is not None else 0} for c, b in buffers.items()]}
        print(f"[REPLAY] Memoria real: {replay_total}/{self.total_a} imágenes A.")

        generator = torch.Generator().manual_seed(cfg["seed"] + 1)
        for r in range(cfg["num_rounds"]):
            self.context = {"phase": 2, "replay_fraction": percent / 100, "round": r + 1}

            def batches(client):
                bx, by = self.train_b[client]
                buf = buffers[client]
                steps = fl_core.steps_per_epoch(len(by), cfg["batch_size"]) * cfg["local_epochs"] if cfg["controlled_replay"] else None
                if cfg["replay_mix"] == "balanced" and buf is not None:
                    return fl_core.mixed_batches(bx, by, buf[0], buf[1], cfg["batch_size"], steps,
                                                 cfg["replay_batch_fraction"], generator)
                return fl_core.concat_batches(bx, by, buf[0] if buf else None, buf[1] if buf else None,
                                              cfg["batch_size"], cfg["local_epochs"], max_steps=steps, generator=generator)

            self.local_round(model, batches, generator, teacher)
            acc_a = self.evaluate(model, *self.test_a)["accuracy_percent"]
            acc_b = self.evaluate(model, *self.test_b)["accuracy_percent"]
            print(f"Fase 2 - Ronda {r + 1}/{cfg['num_rounds']} -> A: {acc_a:.2f}% | B: {acc_b:.2f}%")
            self.history.append({"replay_percent": percent, "round": r + 1,
                                 "accuracy_a_percent": acc_a, "accuracy_b_percent": acc_b})
            with open("history.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
                writer.writeheader()
                writer.writerows(self.history)

        self.context = {"phase": 2, "replay_fraction": percent / 100, "final": True}
        final_a = self.evaluate(model, *self.test_a)
        final_b = self.evaluate(model, *self.test_b)
        if self.args.audit:
            torch.save(model.state_dict(), f"audit_phase2_{percent}.pt")

        extra = {}
        if self.args.validation:
            all_b = (torch.cat([x for x, _ in self.train_b.values()]), torch.cat([y for _, y in self.train_b.values()]))
            extra["train_acc_b"] = fl_core.evaluate(model, *all_b, self.device, self.preprocess)["accuracy_percent"]
            extra["train_acc_replay"] = None
            sets = [b for b in buffers.values() if b is not None]
            if sets:
                xs, ys = torch.cat([b[0] for b in sets]), torch.cat([b[1] for b in sets])
                extra["train_acc_replay"] = fl_core.evaluate(model, xs, ys, self.device, self.preprocess)["accuracy_percent"]
            print(f"[CONTROL] B en entrenamiento: {extra['train_acc_b']:.2f}%; replay: {extra['train_acc_replay']}")

        # Weight Aligning: el mismo modelo con la capa final corregida (no cambia el entrenamiento).
        wa_state, wa_gamma = fl_core.weight_align(model.state_dict(), GROUP_A, GROUP_B)
        wa_model = fl_core.get_model().to(self.device)
        wa_model.load_state_dict(wa_state)
        wa_a = fl_core.evaluate(wa_model, *self.test_a, self.device, self.preprocess, group_a_classes=GROUP_A)
        wa_b = fl_core.evaluate(wa_model, *self.test_b, self.device, self.preprocess)
        print(f"[WA] gamma = {wa_gamma:.3f} -> A: {wa_a['accuracy_percent']:.2f}% | B: {wa_b['accuracy_percent']:.2f}%")
        del teacher, wa_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        acc_a, acc_b = final_a["accuracy_percent"], final_b["accuracy_percent"]
        n_a, n_b = final_a["total"], final_b["total"]
        return {
            "fraction": label,
            "step3_acc_a": self.base_acc_a,
            "step4_acc_a": acc_a,
            "step4_acc_b": acc_b,
            "loss_a": self.base_acc_a - acc_a,
            # Métricas adicionales: no se quedan en el suelo cuando A = 0 %.
            "acc_all_100_classes": (final_a["correct"] + final_b["correct"]) * 100 / (n_a + n_b),
            "retention_a_percent": fl_core.retention_percent(self.base_acc_a, acc_a),
            "group_aware_acc_a": final_a["group_aware_accuracy_percent"],
            "pred_share_b_on_test_a": final_a["pred_share_b_percent"],
            "cross_entropy_a": final_a["mean_cross_entropy"],
            "buffer_samples": replay_total,
            # Métricas estándar de aprendizaje continuo (2 tareas): media por tarea y backward transfer.
            "avg_acc_tasks": (acc_a + acc_b) / 2,
            "bwt_a": acc_a - self.base_acc_a,
            "wa_gamma": wa_gamma,
            "wa_acc_a": wa_a["accuracy_percent"],
            "wa_acc_b": wa_b["accuracy_percent"],
            "wa_acc_all_100_classes": (wa_a["correct"] + wa_b["correct"]) * 100 / (n_a + n_b),
            "wa_retention_a_percent": fl_core.retention_percent(self.base_acc_a, wa_a["accuracy_percent"]),
            "wa_avg_acc_tasks": (wa_a["accuracy_percent"] + wa_b["accuracy_percent"]) / 2,
            "wa_bwt_a": wa_a["accuracy_percent"] - self.base_acc_a,
            **extra,
        }

    def joint(self):
        """Cota superior: A y B a la vez desde cero, con el mismo presupuesto que fase 1 + fase 2.

        Mismas rondas en total (2 × num_rounds) y los mismos pasos por ronda que una
        ronda de fase. No entra en audit.json ni en results.csv: el verificador audita
        solo la secuencia A → B, y esta referencia va a joint_results.csv.
        """
        cfg = self.config
        seed_everything(cfg["seed"])
        rounds = 2 * cfg["num_rounds"]
        print(f"\n=== COTA SUPERIOR: ENTRENAMIENTO CONJUNTO A+B ({rounds} rondas, mismo presupuesto) ===")
        model = fl_core.get_model().to(self.device)
        generator = torch.Generator().manual_seed(cfg["seed"] + 2)
        for r in range(rounds):
            self.context = {"phase": "joint", "round": r + 1}

            def batches(client):
                ax, ay = self.train_a[client]
                bx, by = self.train_b[client]
                steps = fl_core.steps_per_epoch((len(ay) + len(by)) // 2, cfg["batch_size"]) * cfg["local_epochs"]
                return fl_core.concat_batches(ax, ay, bx, by, cfg["batch_size"], cfg["local_epochs"],
                                              max_steps=steps, generator=generator)

            self.local_round(model, batches, generator, audit=False)
            acc_a = fl_core.evaluate(model, *self.test_a, self.device, self.preprocess)["accuracy_percent"]
            acc_b = fl_core.evaluate(model, *self.test_b, self.device, self.preprocess)["accuracy_percent"]
            print(f"Conjunto - Ronda {r + 1}/{rounds} -> A: {acc_a:.2f}% | B: {acc_b:.2f}%")
        final_a = fl_core.evaluate(model, *self.test_a, self.device, self.preprocess, group_a_classes=GROUP_A)
        final_b = fl_core.evaluate(model, *self.test_b, self.device, self.preprocess)
        acc_a, acc_b = final_a["accuracy_percent"], final_b["accuracy_percent"]
        return {"fraction": "Conjunto A+B", "rounds": rounds, "acc_a": acc_a, "acc_b": acc_b,
                "acc_all_100_classes": (final_a["correct"] + final_b["correct"]) * 100 / (final_a["total"] + final_b["total"]),
                "avg_acc_tasks": (acc_a + acc_b) / 2}

    # ---------------------------------------------------------- ejecución
    def run(self):
        args, cfg = self.args, self.config
        self.save_status()
        self.load_data()
        self.phase1()
        results = []
        for percent in args.buffers:
            result = self.phase2(percent)
            results.append(result)
            self.save_audit()
            if args.validation:
                write_csv("diagnostic_results.csv", results)
                problem = learning_problem(result["step4_acc_b"], result["train_acc_b"], result["train_acc_replay"], cfg)
                if problem:
                    self.status["reason"] = f"{problem} con buffer {percent}%; comparación no validada"
                    self.save_status()
                    raise RuntimeError(self.status["reason"])
        joint = None
        if args.joint:
            joint = self.joint()
            write_csv("joint_results.csv", [joint])
            for r in results:
                r["gap_all_vs_joint"] = joint["acc_all_100_classes"] - r["acc_all_100_classes"]
                r["wa_gap_all_vs_joint"] = joint["acc_all_100_classes"] - r["wa_acc_all_100_classes"]
                r["intransigence_b"] = joint["acc_b"] - r["step4_acc_b"]
        self.status.update(
            comparison_valid=args.validation,
            reason=("Controles mínimos del piloto superados; no sustituye varias semillas ni el experimento completo"
                    if args.validation else "Prueba funcional sin validación de aprendizaje" if args.quick
                    else "Ejecución completa; revisar aprendizaje y replicar con varias semillas"))
        self.save_status()
        write_csv("results.csv", results)
        print_table(results, joint)
        if args.quick:
            print("[INFO] Prueba funcional: se omiten los gráficos comparativos porque no validan aprendizaje.")
        else:
            rows = [{"fraction": r["fraction"], "loss_a_pp": r["loss_a"]} for r in results]
            plot_loss([row for row, p in zip(rows, args.buffers) if p <= 20], "group_a_accuracy_loss_up_to_20.png",
                      "Pérdida de precisión en A (buffers hasta 20 %)", xlabel="Método")
            plot_loss(rows, "group_a_accuracy_loss_all_buffers.png", "Pérdida de precisión en A", xlabel="Método")
        if args.audit:
            self.save_audit()
            print("[AUDIT] Diagnósticos guardados en audit.json y pesos en audit_phase*.pt")
        return results


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(results, joint=None):
    width = 146
    print("\n" + "=" * width)
    print(f"{'Método':<14} | {'A antes':>8} | {'A después':>9} | {'B después':>9} | {'Pérdida A':>10} | "
          f"{'Retención':>9} | {'A (grupo)':>9} | {'Pred. B en A':>12} | {'100 clases':>10} | "
          f"{'A con WA':>8} | {'100 cl. WA':>10}")
    print("-" * width)
    for r in results:
        retention = "—" if r["retention_a_percent"] is None else f"{r['retention_a_percent']:.1f}%"
        print(f"{r['fraction']:<14} | {r['step3_acc_a']:>7.2f}% | {r['step4_acc_a']:>8.2f}% | {r['step4_acc_b']:>8.2f}% | "
              f"{r['loss_a']:>7.2f} pp | {retention:>9} | {r['group_aware_acc_a']:>8.2f}% | "
              f"{r['pred_share_b_on_test_a']:>11.1f}% | {r['acc_all_100_classes']:>9.2f}% | "
              f"{r['wa_acc_a']:>7.2f}% | {r['wa_acc_all_100_classes']:>9.2f}%")
    if joint:
        print("-" * width)
        print(f"{joint['fraction']:<14} | {'—':>8} | {joint['acc_a']:>8.2f}% | {joint['acc_b']:>8.2f}% | {'':>10} | "
              f"{'':>9} | {'':>9} | {'':>12} | {joint['acc_all_100_classes']:>9.2f}% | {'':>8} | {'':>10}")
    print("=" * width)
    print("Pérdida A = A antes − A después (pp). «A (grupo)» restringe la predicción a las clases A (diagnóstico).")
    print("WA = Weight Aligning: el mismo modelo con la capa final reescalada para no favorecer a B.")
    if joint:
        print("Conjunto A+B = cota superior: A y B a la vez desde cero, con el mismo presupuesto de rondas y pasos.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config, contract = resolve_config(args, parser)
    if not args.output_dir:
        args.output_dir = os.path.join("resultados", f"cifar100_three_buffer_seed{config['seed']}_{datetime.now():%Y-%m-%d_%H%M%S}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    print(f"[INFO] Resultados en: {os.getcwd()}")
    if args.quick:
        print("QUICK TEST: muestras reducidas; las precisiones no son resultados del experimento completo.")
    if args.validation:
        print("VALIDATION: piloto con aprendizaje mínimo, clases cubiertas y entrenamiento controlado.")
    Path("config.json").write_text(json.dumps({
        **config, "quick": args.quick, "validation": args.validation, "audit": args.audit, "buffers": args.buffers,
        "phase1_training_contract": contract, "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "environment": environment_info(),
        "phase1_checkpoint": str(args.phase1_checkpoint) if args.phase1_checkpoint else None,
        "phase1_checkpoint_sha256": hashlib.sha256(args.phase1_checkpoint.read_bytes()).hexdigest() if args.phase1_checkpoint else None,
    }, indent=2), encoding="utf-8")
    seed_everything(config["seed"])
    print(f"Device in use: {config['device']}")
    return Experiment(args, config, contract).run()


if __name__ == "__main__":
    main()
