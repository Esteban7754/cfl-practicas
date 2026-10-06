"""Control local del muestreo; no sustituye la comparación federada ni usa test para entrenar."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from flwr_datasets import FederatedDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from cifar100_sampling import balanced_order
from verificar_cifar100 import evaluate, new_model, tensors

config = json.loads((ROOT / "config.json").read_text())
torch.set_num_threads(4)
source = ROOT / "audit_phase2_20.pt"
source_weights = torch.load(source, map_location="cpu", weights_only=True)
output = ROOT / "diagnostico_muestreo"
output.mkdir(exist_ok=False)
fds = FederatedDataset(dataset="uoft-cs/cifar100", revision=config["dataset_revision"],
                       partitioners={"train": config["num_clients"]}, seed=config["seed"])
partition = fds.load_partition(0, "train")
group_a = partition.filter(lambda row: row["fine_label"] < 50)
group_b = partition.filter(lambda row: row["fine_label"] >= 50)
memory_indices = balanced_order(group_a["fine_label"], config["seed"])[:len(group_a) // 5]
memory = group_a.select(memory_indices)
b_images, b_labels = tensors(group_b)
a_images, a_labels = tensors(memory)
images = torch.cat((b_images, a_images))
labels = torch.cat((b_labels, a_labels))
test = load_dataset("uoft-cs/cifar100", revision=config["dataset_revision"], split="test")
tests = {name: tensors(test.filter(lambda row: lower <= row["fine_label"] < upper))
         for name, lower, upper in (("A", 0, 50), ("B", 50, 100))}
steps = math.ceil(len(group_b) / config["batch_size"]) * config["local_epochs"]
result = {
    "scope": "Diagnóstico de un cliente desde el modelo federado final con 20%; sin agregación ni ajuste con test",
    "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "config": {"client": 0, "seed": config["seed"], "batch_size": config["batch_size"],
               "optimizer_steps": steps, "lr": config["lr"], "memory_samples": len(memory),
               "new_b_samples": len(group_b)}, "variants": {},
}
for mode in ("mezcla_actual", "lotes_equilibrados"):
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = new_model()
    model.load_state_dict(source_weights)
    before = model.fc.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    order = torch.randperm(len(labels), generator=generator)
    losses = []
    a_seen = b_seen = 0
    for step in range(steps):
        if mode == "mezcla_actual":
            batch_indices = order[step * config["batch_size"]:(step + 1) * config["batch_size"]]
        else:
            half = config["batch_size"] // 2
            old = torch.randint(len(a_labels), (half,), generator=generator) + len(b_labels)
            new = torch.randint(len(b_labels), (config["batch_size"] - half,), generator=generator)
            batch_indices = torch.cat((new, old))
            batch_indices = batch_indices[torch.randperm(len(batch_indices), generator=generator)]
        batch_labels = labels[batch_indices]
        a_seen += (batch_labels < 50).sum().item()
        b_seen += (batch_labels >= 50).sum().item()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images[batch_indices]), batch_labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 20 == 0:
            print(f"{mode}: {step + 1}/{steps} actualizaciones", flush=True)
    metrics = {name: evaluate(model, *xy) for name, xy in tests.items()}
    result["variants"][mode] = {"optimizer_steps": len(losses), "a_samples_processed": a_seen,
                                "b_samples_processed": b_seen, "batch_losses": losses,
                                "fc_weight_delta_l2": (model.fc.weight.detach() - before).norm().item(),
                                "test": metrics}
    torch.save(model.state_dict(), output / f"{mode}.pt")
    (output / "diagnostico.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(mode, json.dumps(metrics), flush=True)
print("CONTROL LOCAL TERMINADO; estos valores no son resultados federados.", flush=True)
