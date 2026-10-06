"""Cuenta los conjuntos completos y contrasta el checkpoint con el registro original."""

import hashlib
import json
import sys
from pathlib import Path

import torch
from flwr_datasets import FederatedDataset

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from verificar_cifar100 import new_model, tensors, evaluate

torch.set_num_threads(4)
fds = FederatedDataset(dataset="uoft-cs/cifar100", revision="aadb3af77e9048adbea6b47c21a81e47dd092ae5",
                       partitioners={"train": 10}, seed=42)
train, test = fds.load_split("train"), fds.load_split("test")
result = {"train_total": len(train), "test_total": len(test),
          "train_a": sum(label < 50 for label in train["fine_label"]),
          "train_b": sum(label >= 50 for label in train["fine_label"]),
          "previous_pilot_a_pool": 1000, "previous_pilot_replay_images": 200}
result["previous_pilot_effective_fraction_of_full_a"] = 200 / result["train_a"]
test_a = test.filter(lambda row: row["fine_label"] < 50)
test_b = test.filter(lambda row: row["fine_label"] >= 50)
result.update(test_a=len(test_a), test_b=len(test_b))
model = new_model()
checkpoint = root / "global_model_phase1.pt"
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
result["full_test_a"] = evaluate(model, *tensors(test_a))
print(json.dumps(result, indent=2), flush=True)
(Path(__file__).resolve().parent / "checkpoint_inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
