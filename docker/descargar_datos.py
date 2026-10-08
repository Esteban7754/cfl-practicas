"""Descarga CIFAR-100 (revisión fijada) en la caché de la imagen y comprueba que funciona sin red.

Se ejecuta durante «docker build». Si la carga sin conexión falla, la construcción
falla también: así nunca se publica una imagen que dependa de Internet en silencio.
"""

import os
import subprocess
import sys

DATASET = "uoft-cs/cifar100"
REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"

CHECK = f"""
from datasets import load_dataset
from flwr_datasets import FederatedDataset
fds = FederatedDataset(dataset={DATASET!r}, revision={REVISION!r}, partitioners={{"train": 10}}, seed=42)
assert len(fds.load_split("test")) == 10000
assert len(fds.load_partition(0, "train")) > 0
for split, size in (("train", 50000), ("test", 10000)):
    assert len(load_dataset({DATASET!r}, revision={REVISION!r}, split=split)) == size
print("CIFAR-100 disponible sin conexión")
"""


def main():
    from datasets import load_dataset

    for split in ("train", "test"):
        data = load_dataset(DATASET, revision=REVISION, split=split)
        print(f"[DATOS] {DATASET}@{REVISION[:8]} {split}: {len(data)} imágenes")
    env = {**os.environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    subprocess.run([sys.executable, "-c", CHECK], check=True, env=env)


if __name__ == "__main__":
    main()
