"""Cuenta imágenes idénticas en los splits oficiales, sin entrenar ni modificar datos."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "config.json").read_text())
dataset = load_dataset("uoft-cs/cifar100", revision=config["dataset_revision"])
indices = {}
for split in ("train", "test"):
    hashes = defaultdict(list)
    for index, row in enumerate(dataset[split]):
        digest = hashlib.sha256(np.asarray(row["img"]).tobytes()).hexdigest()
        hashes[digest].append({"index": index, "label": int(row["fine_label"])})
    indices[split] = hashes
shared = sorted(set(indices["train"]) & set(indices["test"]))
result = {
    "dataset_revision": config["dataset_revision"],
    "official_split_sizes": {split: len(dataset[split]) for split in indices},
    "unique_shared_image_hashes": len(shared),
    "test_samples_identical_to_training": sum(len(indices["test"][digest]) for digest in shared),
    "shared_images": [{"sha256": digest, "train": indices["train"][digest],
                       "test": indices["test"][digest]} for digest in shared],
}
(ROOT / "solapamiento_oficial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "shared_images"}), flush=True)
