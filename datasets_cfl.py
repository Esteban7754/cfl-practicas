"""Especificación de los datasets del protocolo de varias tareas (cfl_tareas.py).

Cada dataset declara de dónde se carga (Hugging Face, revisión fija), cómo se
leen sus columnas, sus estadísticas de normalización y en cuántas tareas se
divide. Añadir otro dataset (p. ej. Tiny-ImageNet) es añadir una entrada a
``DATASETS`` sin tocar el experimento.

No importa nada pesado al cargarse.
"""

import os
from dataclasses import asdict, dataclass

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)  # las mismas que fl_core.CIFAR100_MEAN
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    revision: str
    image_column: str
    label_column: str
    num_classes: int
    num_tasks: int
    image_size: int
    mean: tuple
    std: tuple

    def tasks(self):
        """Clases de cada tarea, en orden fijo: tarea t = clases [t·k, (t+1)·k)."""
        if self.num_classes % self.num_tasks:
            raise ValueError(f"{self.num_classes} clases no se reparten en {self.num_tasks} tareas iguales")
        k = self.num_classes // self.num_tasks
        return [list(range(t * k, (t + 1) * k)) for t in range(self.num_tasks)]

    def describe(self):
        return asdict(self)


DATASETS = {
    # Etiquetas en orden alfabético: el orden 0-99 ya mezcla las superclases.
    "cifar100": DatasetSpec(name="cifar100", hf_id="uoft-cs/cifar100",
                            revision="aadb3af77e9048adbea6b47c21a81e47dd092ae5",
                            image_column="img", label_column="fine_label", num_classes=100, num_tasks=5,
                            image_size=32, mean=CIFAR100_MEAN, std=CIFAR100_STD),
}


def get_spec(name):
    if name not in DATASETS:
        raise ValueError(f"Dataset desconocido: {name}. Disponibles: {', '.join(DATASETS)}")
    return DATASETS[name]


def use_preloaded_data():
    """Con los datos dentro de la imagen Docker, no depende de Internet (salvo que se pida explícitamente)."""
    if os.environ.get("CFL_DATA_PRELOADED") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
