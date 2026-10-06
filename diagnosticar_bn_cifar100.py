"""Contrasta la evaluación del checkpoint con una recalibración de BatchNorm."""

import argparse
import json
from pathlib import Path

import torch
from flwr_datasets import FederatedDataset

from cifar100_sampling import balanced_subset, balanced_order
from verificar_cifar100 import tensors, new_model, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    torch.set_num_threads(4)
    fds = FederatedDataset(dataset="uoft-cs/cifar100", revision=config["dataset_revision"],
                           partitioners={"train": config["num_clients"]}, seed=config["seed"])

    def group(dataset, lower, upper, count):
        selected = dataset.filter(lambda row: lower <= row["fine_label"] < upper)
        return balanced_subset(selected, count, config["seed"])

    test = fds.load_split("test")
    test_tensors = {name: tensors(group(test, lower, upper, config["test_samples_per_group"]))
                    for name, lower, upper in (("A", 0, 50), ("B", 50, 100))}
    train = {name: [] for name in ("A", "B")}
    for client in range(config["num_clients"]):
        partition = fds.load_partition(client, "train")
        for name, lower, upper in (("A", 0, 50), ("B", 50, 100)):
            train[name].append(group(partition, lower, upper, config["train_samples_per_group"]))

    diagnostics = {}
    for percent in config["buffers"]:
        xs, ys = [], []
        replay_xs, replay_ys = [], []
        for client in range(config["num_clients"]):
            x, y = tensors(train["B"][client])
            xs.append(x)
            ys.append(y)
            buffer = train["A"][client]
            indices = balanced_order(buffer["fine_label"], config["seed"] + client)
            size = len(buffer) * percent // 100
            if size:
                x, y = tensors(buffer.select(indices[:size]))
                xs.append(x)
                ys.append(y)
                replay_xs.append(x)
                replay_ys.append(y)
        images, labels = torch.cat(xs), torch.cat(ys)
        model = new_model()
        model.load_state_dict(torch.load(args.run_dir / f"audit_phase2_{percent}.pt", map_location="cpu", weights_only=True))
        before = {name: evaluate(model, *xy) for name, xy in test_tensors.items()}
        before["training_mixture"] = evaluate(model, images, labels)
        if replay_xs:
            before["replay_training"] = evaluate(model, torch.cat(replay_xs), torch.cat(replay_ys))

        # Se cambian SOLO estadísticas de BatchNorm. No hay actualizaciones de pesos
        # ni imágenes de test en la calibración. Ambos buffers reciben el mismo método.
        weights_before = {name: value.detach().clone() for name, value in model.named_parameters()}
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.reset_running_stats()
                layer.momentum = None
        model.train()
        order = torch.randperm(len(images), generator=torch.Generator().manual_seed(config["seed"]))
        with torch.no_grad():
            for batch in images[order].split(32):
                model(batch)
        assert all(torch.equal(value, weights_before[name]) for name, value in model.named_parameters())
        after = {name: evaluate(model, *xy) for name, xy in test_tensors.items()}
        after["training_mixture"] = evaluate(model, images, labels)
        if replay_xs:
            after["replay_training"] = evaluate(model, torch.cat(replay_xs), torch.cat(replay_ys))
        diagnostics[str(percent)] = {"before": before, "after_bn_recalibration": after,
                                     "trainable_parameters_unchanged": True}
        print(percent, json.dumps(diagnostics[str(percent)]), flush=True)
    (args.run_dir / "batchnorm_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
