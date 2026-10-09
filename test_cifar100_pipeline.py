"""Prueba de extremo a extremo de cifar100_three_buffer.py con datos sintéticos.

Sustituye ``FederatedDataset`` por un conjunto pequeño generado en memoria
(sin Internet) y comprueba que el flujo completo funciona con las variantes de
replay: concat, lotes equilibrados y destilación. No evalúa la calidad del
aprendizaje: solo que el código es coherente y deja los artefactos esperados.

Requiere torch, datasets y flwr-datasets (se omite si faltan).
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import datasets
    import flwr_datasets  # noqa: F401
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    datasets = None


def synthetic_split(n_per_class, seed, classes=range(100)):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in classes:
        base = np.zeros((32, 32, 3), dtype=np.uint8)
        base[(label % 8) * 4:(label % 8) * 4 + 4, :, label % 3] = 200  # patrón por clase
        base[:, (label // 8) * 2:(label // 8) * 2 + 2, (label + 1) % 3] = 150
        for _ in range(n_per_class):
            noise = rng.integers(0, 40, size=(32, 32, 3), dtype=np.uint8)
            images.append(np.clip(base + noise, 0, 255).astype(np.uint8))
            labels.append(label)
    features = datasets.Features({"img": datasets.Image(), "fine_label": datasets.Value("int64")})
    return datasets.Dataset.from_dict({"img": images, "fine_label": labels}, features=features)


class FakeFederatedDataset:
    def __init__(self, dataset, revision, partitioners, seed):
        self.num = partitioners["train"]
        self.train = synthetic_split(6, seed).shuffle(seed=seed)
        self.test = synthetic_split(2, seed + 1).shuffle(seed=seed)

    def load_split(self, split):
        return self.test

    def load_partition(self, partition_id, split="train"):
        return self.train.shard(num_shards=self.num, index=partition_id)


@unittest.skipIf(datasets is None, "requiere torch, datasets y flwr-datasets")
class PipelineTests(unittest.TestCase):
    def run_experiment(self, *extra):
        import cifar100_three_buffer

        cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            with mock.patch("flwr_datasets.FederatedDataset", FakeFederatedDataset):
                results = cifar100_three_buffer.main(
                    ["--quick", "--audit", "--buffers", "0", "20", "--output-dir", tmp, *extra])
        finally:
            os.chdir(cwd)
        return results, Path(tmp)

    def test_concat_replay_produces_consistent_artifacts(self):
        results, out = self.run_experiment()
        self.assertEqual([r["fraction"] for r in results], ["No Replay", "20% Replay"])
        for name in ("config.json", "results.csv", "audit.json", "validation.json", "data_manifest.json",
                     "audit_phase1.pt", "audit_phase2_0.pt", "audit_phase2_20.pt", "history.csv"):
            self.assertTrue((out / name).exists(), name)
        audit = json.loads((out / "audit.json").read_text())
        # Mismos pesos iniciales en todas las variantes.
        self.assertEqual(len(set(audit["phase2_initial_weights"].values())), 1)
        self.assertEqual(audit["phase1_weights_sha256"], next(iter(audit["phase2_initial_weights"].values())))
        # Mismo número de actualizaciones por cliente y ronda en todas las variantes de la fase 2.
        phase2 = [row for row in audit["training"] if row["context"]["phase"] == 2]
        steps = {}
        for row in phase2:
            steps.setdefault((row["context"]["client"], row["context"]["round"]), set()).add(row["optimizer_steps"])
        self.assertTrue(all(len(values) == 1 for values in steps.values()), steps)
        # El replay usa muestras A; sin replay no.
        a_seen = {round(row["context"]["replay_fraction"] * 100): sum(v for k, v in row["label_counts"].items() if int(k) < 50)
                  for row in phase2}
        self.assertEqual(a_seen[0], 0)
        self.assertGreater(a_seen[20], 0)
        with open(out / "results.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertAlmostEqual(float(row["loss_a"]), float(row["step3_acc_a"]) - float(row["step4_acc_a"]))
            self.assertIn("group_aware_acc_a", row)
        config = json.loads((out / "config.json").read_text())
        self.assertTrue(config["normalize"] and config["augment"] and config["controlled_replay"])

    def test_balanced_batches_and_distillation_run(self):
        results, out = self.run_experiment("--replay-mix", "balanced", "--replay-batch-fraction", "0.5",
                                           "--distill-weight", "1.0")
        audit = json.loads((out / "audit.json").read_text())
        rows = [r for r in audit["training"] if r["context"]["phase"] == 2 and r["context"]["replay_fraction"] > 0]
        for row in rows:
            a = sum(v for k, v in row["label_counts"].items() if int(k) < 50)
            self.assertEqual(a, row["optimizer_steps"] * 16)  # 16 de cada lote de 32 vienen del buffer
        self.assertEqual(len(results), 2)

    def test_joint_baseline_and_weight_aligning(self):
        plain, _ = self.run_experiment()
        results, out = self.run_experiment("--joint")
        # La cota superior va aparte y no altera la secuencia A -> B ni lo que audita el verificador.
        self.assertEqual([r["step4_acc_a"] for r in results], [r["step4_acc_a"] for r in plain])
        audit = json.loads((out / "audit.json").read_text())
        self.assertTrue(all(row["context"]["phase"] in (1, 2) for row in audit["training"]))
        with open(out / "joint_results.csv", encoding="utf-8") as handle:
            joint = list(csv.DictReader(handle))
        self.assertEqual(len(joint), 1)
        self.assertEqual(int(joint[0]["rounds"]), 2)  # --quick: 1 ronda por fase
        with open(out / "results.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertGreater(float(row["wa_gamma"]), 0)
            self.assertAlmostEqual(float(row["avg_acc_tasks"]), (float(row["step4_acc_a"]) + float(row["step4_acc_b"])) / 2)
            self.assertAlmostEqual(float(row["bwt_a"]), -float(row["loss_a"]))
            self.assertAlmostEqual(float(row["gap_all_vs_joint"]),
                                   float(joint[0]["acc_all_100_classes"]) - float(row["acc_all_100_classes"]))

    def test_variant_results_do_not_depend_on_order(self):
        first, _ = self.run_experiment()
        only_20, _ = self.run_experiment("--buffers", "0", "20")
        self.assertEqual(first[1]["step4_acc_a"], only_20[1]["step4_acc_a"])

    def test_old_checkpoint_requires_old_preprocessing(self):
        import cifar100_three_buffer

        parser = cifar100_three_buffer.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp, "ckpt.pt")
            ckpt.write_bytes(b"x")
            Path(tmp, "ckpt.config.json").write_text(json.dumps({
                "num_clients": 10, "seed": 42, "dataset_revision": "aadb3af77e9048adbea6b47c21a81e47dd092ae5"}))
            with self.assertRaises(SystemExit), mock.patch.object(sys, "stderr"):
                cifar100_three_buffer.resolve_config(parser.parse_args(["--phase1-checkpoint", str(ckpt)]), parser)
            config, _ = cifar100_three_buffer.resolve_config(
                parser.parse_args(["--phase1-checkpoint", str(ckpt), "--no-normalize", "--no-augment"]), parser)
            self.assertFalse(config["normalize"])


if __name__ == "__main__":
    unittest.main()
