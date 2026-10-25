"""Aprendizaje federado continuo con varias tareas (CIFAR-100: 5 tareas de 20 clases) y replay.

Protocolo en docs/plan_cinco_tareas.md. Los hiperparámetros están fijos (``FIJOS``) y
se copian a config.json; la línea de órdenes solo expone variables de diseño:

  --metodo concat       mezcla normal: tarea actual y memoria barajadas juntas.
  --metodo balanced     lotes equilibrados con fracción de memoria (t-1)/t en la tarea t.
  --metodo calendario   lotes equilibrados con la fracción por ronda de FIJOS["calendario"].
  --metodo ace          ER-ACE: los datos nuevos solo compiten entre clases nuevas.
  --buffers 0 10 20     memoria fija por cliente: % de las imágenes de una tarea del cliente,
                        repartida a partes iguales entre las clases vistas (0 = sin replay).
  --partition dirichlet reparto desigual por clases (α = FIJOS["dirichlet_alpha"]).
  --quick               prueba funcional mínima; no sirve para comparar.

Cada buffer parte de los mismos pesos tras la primera tarea, con la misma semilla y el mismo
número de pasos por ronda (``épocas · ceil(n_tarea / lote)``), sea cual sea la memoria.
Tras la última tarea se evalúan también WA y cRT sobre el mismo modelo (columnas wa_*, crt_*).

Salidas: results.csv (una fila por buffer), matriz.csv (precisión de cada tarea tras cada tarea),
history.csv (por ronda) y, con --joint, joint_results.csv.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch

import fl_core
from cifar100_sampling import balanced_order
from datasets_cfl import DATASETS, get_spec, use_preloaded_data
from memoria_cfl import ClientMemory, memory_capacity
from reproducibility import environment_info, seed_everything

# Hiperparámetros del protocolo: no se cambian por línea de órdenes (acuerdo del 25/10/2026).
FIJOS = {
    "num_clients": 10,
    "batch_size": 32,
    "local_epochs": 5,
    "rounds_per_task": 5,
    "lr": 0.001,
    "optimizer": "adam",
    "model": "resnet18_cifar",  # conv 3 × 3 sin max-pooling (fl_core.get_model)
    "normalize": True,
    "augment": True,
    "crt_epochs": 2,
    "calendario": [0.75, 0.5, 0.25, 0.25, 0.25],  # una fracción por ronda, igual en cada tarea
    "dirichlet_alpha": 0.5,
}

# Prueba funcional: pocos clientes, rondas y muestras. Solo comprueba que el código funciona.
QUICK = {"num_clients": 2, "local_epochs": 1, "rounds_per_task": 1,
         "train_samples_per_task": 100, "test_samples_per_task": 100}

METODOS = ("concat", "balanced", "calendario", "ace")

# Claves que deben coincidir entre la ejecución y el checkpoint de la primera tarea que reutiliza.
CONTRATO = ("dataset", "dataset_revision", "tasks", "num_clients", "batch_size", "local_epochs", "rounds_per_task",
            "lr", "seed", "partition", "dirichlet_alpha", "normalize", "augment", "train_samples_per_task")


def checkpoint_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint(config, checkpoint, parser):
    """El checkpoint debe venir con su manifiesto, coincidir en CONTRATO y en su SHA-256."""
    manifest_path = checkpoint.with_suffix(".config.json")
    if not manifest_path.is_file():
        parser.error(f"El checkpoint necesita su manifiesto {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    differences = [key for key in CONTRATO if config.get(key) != manifest.get(key)]
    if differences:
        parser.error("El checkpoint de la primera tarea no coincide en: " + ", ".join(differences))
    if manifest.get("checkpoint_sha256") != checkpoint_sha256(checkpoint):
        parser.error("El manifiesto no corresponde al SHA-256 de este checkpoint")
    return manifest


def parse_percent(value):
    percent = int(value)
    if not 0 <= percent < 100:
        raise argparse.ArgumentTypeError("los porcentajes de buffer deben estar entre 0 y 99")
    return percent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodo", choices=METODOS, default="concat", help="Cómo se usa la memoria en cada lote")
    parser.add_argument("--buffers", nargs="+", type=parse_percent, default=[0, 10, 20],
                        help="Tamaño de la memoria por cliente (%% de una tarea); 0 (sin replay) es obligatorio")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del experimento")
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="iid", help="Reparto entre clientes")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar100")
    parser.add_argument("--joint", action="store_true",
                        help="Añade la referencia conjunta: todas las clases a la vez desde cero, mismo presupuesto")
    parser.add_argument("--quick", action="store_true", help="Prueba funcional reducida; no sirve para comparar")
    parser.add_argument("--audit", action="store_true", help="Guarda diagnósticos en audit.json")
    parser.add_argument("--only-phase1", action="store_true",
                        help="Entrena y guarda la primera tarea (phase1_checkpoint.pt) y termina")
    parser.add_argument("--phase1-checkpoint", type=Path, help="Reutiliza los pesos de la primera tarea")
    parser.add_argument("--output-dir", help="Directorio para los resultados (nuevo por defecto)")
    return parser


def resolve_config(args, parser):
    if 0 not in args.buffers or len(args.buffers) != len(set(args.buffers)):
        parser.error("--buffers debe incluir 0 y no repetir porcentajes")
    args.buffers.sort()
    if args.only_phase1 and args.phase1_checkpoint:
        parser.error("--only-phase1 entrena la primera tarea; no se combina con --phase1-checkpoint")
    if args.phase1_checkpoint:
        args.phase1_checkpoint = args.phase1_checkpoint.resolve()
        if not args.phase1_checkpoint.is_file():
            parser.error("No existe el checkpoint de la primera tarea")
    spec = get_spec(args.dataset)
    config = {**FIJOS, **(QUICK if args.quick else {})}
    if len(config["calendario"]) != config["rounds_per_task"]:
        config["calendario"] = config["calendario"][:config["rounds_per_task"]]
    config.update(dataset=spec.name, dataset_revision=spec.revision, num_tasks=spec.num_tasks, tasks=spec.tasks(),
                  metodo=args.metodo, buffers=args.buffers, seed=args.seed, partition=args.partition,
                  dirichlet_alpha=config["dirichlet_alpha"] if args.partition == "dirichlet" else None,
                  joint=args.joint, quick=args.quick,
                  device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if args.phase1_checkpoint:
        config["phase1_manifest"] = validate_checkpoint(config, args.phase1_checkpoint, parser)
    return config, spec


class TaskExperiment:
    def __init__(self, args, config, spec):
        self.args, self.config, self.spec = args, config, spec
        self.device = config["device"]
        self.tasks = config["tasks"]
        self.preprocess = fl_core.Preprocess(normalize=config["normalize"], mean=spec.mean, std=spec.std)
        self.audit = {"training": [], "memory": {}}
        self.context = {}
        self.history, self.matrix = [], []

    # ---------------------------------------------------------- datos
    def split_by_task(self, x, y, limit, seed):
        """Divide (x, y) por tareas; con ``limit`` toma una muestra aleatoria reproducible de cada tarea."""
        out = []
        for t, classes in enumerate(self.tasks):
            idx = torch.nonzero(torch.isin(y, torch.as_tensor(classes))).flatten()
            if limit is not None and len(idx) > limit:
                order = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed + t))
                idx = idx[order[:limit]].sort().values
            out.append((x[idx], y[idx]))
        return out

    def load_data(self):
        """test[t] = (x, y) de la tarea t; train[cliente][t] = (x, y) de la tarea t en ese cliente."""
        use_preloaded_data()
        from flwr_datasets import FederatedDataset

        cfg, spec = self.config, self.spec
        partitioner = cfg["num_clients"]  # IID
        if cfg["partition"] == "dirichlet":
            from flwr_datasets.partitioner import DirichletPartitioner

            partitioner = DirichletPartitioner(num_partitions=cfg["num_clients"], partition_by=spec.label_column,
                                               alpha=cfg["dirichlet_alpha"], seed=cfg["seed"])
        fds = FederatedDataset(dataset=spec.hf_id, revision=spec.revision,
                               partitioners={"train": partitioner}, seed=cfg["seed"])
        to_tensors = lambda part: fl_core.hf_to_tensors(part, spec.image_column, spec.label_column)  # noqa: E731
        self.test = self.split_by_task(*to_tensors(fds.load_split("test")), cfg.get("test_samples_per_task"), cfg["seed"])
        self.train = {}
        for client in range(cfg["num_clients"]):
            partition = to_tensors(fds.load_partition(partition_id=client, split="train"))
            self.train[client] = self.split_by_task(*partition, cfg.get("train_samples_per_task"),
                                                    cfg["seed"] * 1000 + client)
        manifest = {
            "dataset": spec.describe(), "tasks": self.tasks, "partition": cfg["partition"],
            "dirichlet_alpha": cfg["dirichlet_alpha"], "preprocess": self.preprocess.describe(), "augment": cfg["augment"],
            "test": [{"task": t + 1, "samples": len(y), "classes_present": len(y.unique())}
                     for t, (_, y) in enumerate(self.test)],
            "clients": [{"client": c, "tasks": [{"task": t + 1, "samples": len(y), "classes_present": len(y.unique())}
                                                for t, (_, y) in enumerate(tasks)]}
                        for c, tasks in self.train.items()],
            "total_per_task": [sum(len(self.train[c][t][1]) for c in self.train) for t in range(len(self.tasks))],
        }
        Path("data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[DATA] {len(self.tasks)} tareas de {len(self.tasks[0])} clases; "
              f"imágenes de entrenamiento por tarea: {manifest['total_per_task']}")
        return manifest

    # ---------------------------------------------------------- memoria
    def new_memories(self, percent):
        """Memoria vacía de cada cliente con capacidad fija: ``percent`` % de una tarea media del cliente."""
        memories = {}
        for client, tasks in self.train.items():
            samples = sum(len(y) for _, y in tasks)
            memories[client] = ClientMemory(memory_capacity(percent, samples, len(self.tasks)),
                                            self.config["seed"], client, self.spec.num_classes)
        return memories

    # ---------------------------------------------------------- evaluación
    def seen_test(self, t):
        """Test de las tareas 0..t juntas."""
        return torch.cat([x for x, _ in self.test[:t + 1]]), torch.cat([y for _, y in self.test[:t + 1]])

    def evaluate_tasks(self, model, t):
        """Precisión (argmax sobre todas las salidas) en cada tarea vista y en todas juntas tras la tarea t.

        ``pred_new_on_old``: % de predicciones de la tarea t sobre el test de las anteriores (sesgo hacia lo reciente).
        """
        per_task = [fl_core.evaluate(model, x, y, self.device, self.preprocess)["accuracy_percent"]
                    for x, y in self.test[:t + 1]]
        seen = fl_core.evaluate(model, *self.seen_test(t), self.device, self.preprocess)
        out = {"per_task": per_task, "acc_seen": seen["accuracy_percent"], "pred_new_on_old": None}
        if t > 0:
            old = [c for task in self.tasks[:t] for c in task]
            out["pred_new_on_old"] = fl_core.evaluate(model, *self.seen_test(t - 1), self.device, self.preprocess,
                                                      group_a_classes=old)["pred_share_b_percent"]
        return out

    # ---------------------------------------------------------- entrenamiento
    def new_model(self, state=None):
        model = fl_core.get_model(self.spec.num_classes).to(self.device)
        if state is not None:
            model.load_state_dict(state)
        return model

    def local_round(self, global_model, batches_for_client, generator, size_of, ace_new_classes=None,
                    classifier_only=False, audit=True):
        """Una ronda federada. ``size_of(cliente)`` = muestras que aporta; con 0 no entrena ni se promedia.

        En IID FedAvg no pondera (clientes iguales); con Dirichlet pondera por tamaño.
        """
        weights, sizes = [], []
        for client in range(self.config["num_clients"]):
            size = size_of(client)
            if size == 0:
                continue
            self.context["client"] = client
            record = {} if self.args.audit and audit else None
            weights.append(fl_core.train_local(
                self.new_model(global_model.state_dict()), batches_for_client(client), self.config["lr"], self.device,
                self.preprocess, augment=self.config["augment"], generator=generator, record=record,
                ace_new_classes=ace_new_classes, classifier_only=classifier_only))
            sizes.append(size)
            if record is not None:
                record.pop("batch_losses")
                self.audit["training"].append({"context": dict(self.context), **record})
        if weights:
            global_model.load_state_dict(fl_core.fedavg(weights, sizes if self.config["partition"] == "dirichlet" else None))

    def steps(self, client, t):
        """Pasos por ronda en la tarea t: los de recorrer sus datos ``local_epochs`` veces, con o sin memoria."""
        return fl_core.steps_per_epoch(len(self.train[client][t][1]), self.config["batch_size"]) * self.config["local_epochs"]

    def first_task(self):
        """Primera tarea: se entrena una vez y todos los buffers parten de estos pesos."""
        cfg = self.config
        print(f"\n=== TAREA 1/{len(self.tasks)} (común a todos los buffers) ===")
        model = self.new_model()
        if self.args.phase1_checkpoint:
            model.load_state_dict(torch.load(self.args.phase1_checkpoint, map_location=self.device, weights_only=True))
            print(f"[INFO] Primera tarea reutilizada de: {self.args.phase1_checkpoint}")
        else:
            generator = torch.Generator().manual_seed(cfg["seed"])
            for r in range(cfg["rounds_per_task"]):
                self.context = {"task": 1, "round": r + 1}
                self.local_round(model, lambda c: fl_core.concat_batches(
                    *self.train[c][0], None, None, cfg["batch_size"], cfg["local_epochs"], generator=generator),
                    generator, size_of=lambda c: len(self.train[c][0][1]))
                acc = self.evaluate_tasks(model, 0)["acc_seen"]
                print(f"Tarea 1 - Ronda {r + 1}/{cfg['rounds_per_task']} -> {acc:.2f}%")
        self.base_weights = copy.deepcopy(model.state_dict())
        self.base_eval = self.evaluate_tasks(model, 0)
        print(f"[INFO] Precisión en la tarea 1 antes de continuar: {self.base_eval['acc_seen']:.2f}%")
        if self.args.only_phase1:
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        """Guarda los pesos de la primera tarea con su manifiesto, para reutilizarlos con trazabilidad."""
        torch.save(model.state_dict(), "phase1_checkpoint.pt")
        manifest = {key: self.config.get(key) for key in CONTRATO}
        manifest.update(checkpoint_sha256=checkpoint_sha256("phase1_checkpoint.pt"),
                        provenance=f"Generado por cfl_tareas.py el {datetime.now():%Y-%m-%d %H:%M}",
                        task1_test_accuracy_percent=self.base_eval["acc_seen"], torch_version=torch.__version__)
        Path("phase1_checkpoint.config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def batches(self, method, client, t, r, memory, generator):
        """Lotes de un cliente en la ronda r de la tarea t (desde 0) según el método."""
        cfg = self.config
        x, y = self.train[client][t]
        steps = self.steps(client, t)
        if memory is None:  # sin replay
            return fl_core.concat_batches(x, y, None, None, cfg["batch_size"], cfg["local_epochs"], max_steps=steps,
                                          generator=generator)
        if method == "concat":
            return fl_core.concat_batches(x, y, *memory, cfg["batch_size"], cfg["local_epochs"], max_steps=steps,
                                          generator=generator)
        fraction = cfg["calendario"][r] if method == "calendario" else t / (t + 1)  # antiguas / vistas
        if method == "ace":
            return fl_core.mixed_batches(x, y, *memory, cfg["batch_size"], steps, fraction, generator, with_origin=True)
        return fl_core.mixed_batches(x, y, *memory, cfg["batch_size"], steps, fraction, generator)

    def ace_memory(self, memory, client, t):
        """Memoria de ER-ACE: la memoria de clases antiguas más otras tantas imágenes de la tarea actual.

        En ER-ACE la memoria también guarda la tarea actual; sin ella, nada empuja las clases nuevas por
        encima de las antiguas. Esas imágenes ya están en el cliente: no cuesta memoria de tareas pasadas.
        """
        x, y = self.train[client][t]
        if len(y) == 0:  # Dirichlet: el cliente no tiene esta tarea y no entrena
            return memory
        idx = torch.as_tensor(balanced_order(y.tolist(), self.config["seed"] + 7 + client)[:len(memory[1])])
        return torch.cat([memory[0], x[idx]]), torch.cat([memory[1], y[idx]])

    def record_task(self, percent, t, result):
        for i, acc in enumerate(result["per_task"]):
            self.matrix.append({"buffer": percent, "tras_tarea": t + 1, "tarea": i + 1, "acc": acc})
        write_csv("matriz.csv", self.matrix)

    def run_buffer(self, percent):
        """Tareas 2..T con una memoria de ``percent`` %; devuelve la fila de results.csv."""
        cfg, method, last = self.config, self.config["metodo"], len(self.tasks) - 1
        seed_everything(cfg["seed"])
        print(f"\n=== BUFFER {percent} % ({method}) ===")
        model = self.new_model(copy.deepcopy(self.base_weights))
        memories = self.new_memories(percent)
        for client, memory in memories.items():
            memory.add_task(*self.train[client][0])
        evals = [self.base_eval]
        self.record_task(percent, 0, self.base_eval)
        generator = torch.Generator().manual_seed(cfg["seed"] + 1)
        for t in range(1, len(self.tasks)):
            stored = {c: m.tensors() for c, m in memories.items()}
            ace = method == "ace" and percent > 0  # sin memoria, ER-ACE es entropía cruzada normal
            if ace:
                stored = {c: None if m is None else self.ace_memory(m, c, t) for c, m in stored.items()}
            for r in range(cfg["rounds_per_task"]):
                self.context = {"buffer": percent, "task": t + 1, "round": r + 1}
                self.local_round(model, lambda c: self.batches(method, c, t, r, stored[c], generator), generator,
                                 size_of=lambda c: len(self.train[c][t][1]),
                                 ace_new_classes=self.tasks[t] if ace else None)
                result = self.evaluate_tasks(model, t)
                print(f"Tarea {t + 1} - Ronda {r + 1}/{cfg['rounds_per_task']} -> vistas {result['acc_seen']:.2f}% | "
                      + " ".join(f"T{i + 1} {a:.1f}" for i, a in enumerate(result["per_task"])))
                self.history.append({"buffer": percent, "tarea": t + 1, "ronda": r + 1, "acc_vistas": result["acc_seen"],
                                     **{f"acc_t{i + 1}": a for i, a in enumerate(result["per_task"])}})
                write_csv("history.csv", self.history)
            evals.append(result)
            self.record_task(percent, t, result)
            if t < last:  # la memoria tras la última tarea ya no se usa para entrenar
                for client, memory in memories.items():
                    memory.add_task(*self.train[client][t])
        self.audit["memory"][str(percent)] = [{"client": c, "capacity": m.capacity, "size": m.size()}
                                              for c, m in memories.items()]
        row = self.summary(percent, evals, memories)
        row.update(self.post_hoc(model, memories, percent))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return row

    def summary(self, percent, evals, memories):
        """Métricas de aprendizaje continuo a partir de las evaluaciones tras cada tarea."""
        last = len(evals) - 1
        R = [e["per_task"] for e in evals]
        old = range(last)
        quotas = [n for m in memories.values() for n in m.quotas().values()]
        return {
            "buffer": percent, "metodo": self.config["metodo"],
            "acc_final": evals[-1]["acc_seen"],
            "acc_incremental_media": sum(e["acc_seen"] for e in evals) / len(evals),
            "bwt": sum(R[last][i] - R[i][i] for i in old) / last,
            "olvido_medio": sum(max(R[t][i] for t in range(i, last)) - R[last][i] for i in old) / last,
            "acc_t1_final": R[last][0], f"acc_t{last + 1}_final": R[last][last],
            "acc_t1_inicial": R[0][0],
            "pred_nuevas_en_antiguas": evals[-1]["pred_new_on_old"],
            "memoria_total": sum(m.size() for m in memories.values()),
            "memoria_por_clase_final": sum(quotas) / len(quotas) if quotas else 0,
        }

    def post_hoc(self, model, memories, percent):
        """Correcciones del clasificador sobre el modelo final; no cambian el entrenamiento.

        WA: antiguas = tareas 1..T-1, nuevas = tarea T. cRT: la capa final reentrenada (resto congelado)
        con la memoria y el mismo número de imágenes de la última tarea en cada cliente.
        """
        cfg, last, out = self.config, len(self.tasks) - 1, {}

        def summary(prefix, state):
            result = self.evaluate_tasks(self.new_model(state), last)
            out.update({f"{prefix}acc_final": result["acc_seen"], f"{prefix}acc_t1_final": result["per_task"][0],
                        f"{prefix}acc_t{last + 1}_final": result["per_task"][last]})

        old = [c for task in self.tasks[:last] for c in task]
        aligned, out["wa_gamma"] = fl_core.weight_align(model.state_dict(), old, self.tasks[last])
        summary("wa_", aligned)
        print(f"[WA] gamma = {out['wa_gamma']:.3f} -> final {out['wa_acc_final']:.2f}%")
        if percent > 0 and cfg["crt_epochs"] > 0:
            crt = self.new_model(model.state_dict())
            generator = torch.Generator().manual_seed(cfg["seed"] + 3)

            def balanced_set(client):
                stored, (x, y) = memories[client].tensors(), self.train[client][last]
                if stored is None or len(y) == 0:
                    return None
                idx = torch.randperm(len(y), generator=generator)[:len(stored[1])]
                return torch.cat([stored[0], x[idx]]), torch.cat([stored[1], y[idx]])

            sets = {c: balanced_set(c) for c in range(cfg["num_clients"])}
            self.local_round(crt, lambda c: fl_core.concat_batches(*sets[c], None, None, cfg["batch_size"],
                                                                   cfg["crt_epochs"], generator=generator),
                             generator, size_of=lambda c: 0 if sets[c] is None else len(sets[c][1]),
                             classifier_only=True, audit=False)
            summary("crt_", crt.state_dict())
            print(f"[cRT] final {out['crt_acc_final']:.2f}%")
        return out

    def joint(self):
        """Referencia conjunta: todas las clases a la vez desde cero, con el mismo presupuesto.

        Mismas rondas en total (tareas × rondas por tarea) y, en cada ronda, los pasos de una tarea media.
        """
        cfg = self.config
        seed_everything(cfg["seed"])
        rounds = len(self.tasks) * cfg["rounds_per_task"]
        print(f"\n=== REFERENCIA: TODAS LAS CLASES A LA VEZ ({rounds} rondas) ===")
        data = {c: (torch.cat([x for x, _ in tasks]), torch.cat([y for _, y in tasks])) for c, tasks in self.train.items()}
        model = self.new_model()
        generator = torch.Generator().manual_seed(cfg["seed"] + 2)
        for r in range(rounds):
            self.context = {"phase": "joint", "round": r + 1}
            self.local_round(model, lambda c: fl_core.concat_batches(
                *data[c], None, None, cfg["batch_size"], cfg["local_epochs"],
                max_steps=fl_core.steps_per_epoch(len(data[c][1]) // len(self.tasks), cfg["batch_size"]) * cfg["local_epochs"],
                generator=generator), generator, size_of=lambda c: len(data[c][1]), audit=False)
        result = self.evaluate_tasks(model, len(self.tasks) - 1)
        print(f"[CONJUNTO] {result['acc_seen']:.2f}%")
        return {"rounds": rounds, "acc_final": result["acc_seen"],
                **{f"acc_t{i + 1}": a for i, a in enumerate(result["per_task"])}}

    # ---------------------------------------------------------- ejecución
    def run(self):
        self.load_data()
        self.first_task()
        if self.args.only_phase1:
            print("[INFO] Primera tarea guardada en phase1_checkpoint.pt; no se continúa.")
            return []
        results = [self.run_buffer(percent) for percent in self.config["buffers"]]
        if self.args.joint:
            joint = self.joint()
            write_csv("joint_results.csv", [joint])
            for row in results:
                row["gap_vs_joint"] = joint["acc_final"] - row["acc_final"]
        write_csv("results.csv", results)
        if self.args.audit:
            Path("audit.json").write_text(json.dumps(self.audit, indent=2), encoding="utf-8")
        print_table(results)
        return results


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(results):
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    last = next(k for k in results[0] if k.startswith("acc_t") and k.endswith("_final") and k != "acc_t1_final")
    print(f"\n{'Buffer':>7} | {'Final':>6} | {'Incr.':>6} | {'BWT':>7} | {'Olvido':>6} | {'T1':>6} | {'Últ.':>6} | "
          f"{'WA':>6} | {'cRT':>6}")
    for r in results:
        print(f"{r['buffer']:>6}% | {fmt(r['acc_final']):>6} | {fmt(r['acc_incremental_media']):>6} | {fmt(r['bwt']):>7} | "
              f"{fmt(r['olvido_medio']):>6} | {fmt(r['acc_t1_final']):>6} | {fmt(r[last]):>6} | "
              f"{fmt(r['wa_acc_final']):>6} | {fmt(r.get('crt_acc_final')):>6}")
    print("Final = todas las clases tras la última tarea; Incr. = media tras cada tarea; Últ. = última tarea.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config, spec = resolve_config(args, parser)
    if not args.output_dir:
        args.output_dir = os.path.join("resultados", f"tareas_{config['metodo']}_seed{config['seed']}_{datetime.now():%Y-%m-%d_%H%M%S}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    print(f"[INFO] Resultados en: {os.getcwd()}")
    if args.quick:
        print("QUICK TEST: muestras reducidas; las precisiones no son resultados del experimento.")
    Path("config.json").write_text(json.dumps({
        **config, "fijos": FIJOS, "audit": args.audit, "only_phase1": args.only_phase1,
        "phase1_checkpoint": str(args.phase1_checkpoint) if args.phase1_checkpoint else None,
        "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(), "environment": environment_info(),
    }, indent=2), encoding="utf-8")
    seed_everything(config["seed"])
    print(f"Device in use: {config['device']}")
    return TaskExperiment(args, config, spec).run()


if __name__ == "__main__":
    main()
