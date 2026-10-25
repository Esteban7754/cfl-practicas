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

Estado: fase 1 del plan (datos por tareas y configuración). El bucle de tareas llega en la fase 3.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

import fl_core
from datasets_cfl import DATASETS, get_spec, use_preloaded_data
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
    return config, spec


class TaskExperiment:
    def __init__(self, args, config, spec):
        self.args, self.config, self.spec = args, config, spec
        self.device = config["device"]
        self.tasks = config["tasks"]
        self.preprocess = fl_core.Preprocess(normalize=config["normalize"], mean=spec.mean, std=spec.std)

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

    # ---------------------------------------------------------- ejecución
    def run(self):
        self.load_data()
        print("[INFO] Fase 1 del plan: datos y configuración listos. El bucle de tareas llega en la fase 3.")
        return []


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
