"""Tests de cfl_tareas.py (protocolo de varias tareas) con datos sintéticos y sin red.

Reutiliza el ``FakeFederatedDataset`` de test_cifar100_pipeline.py.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets_cfl import get_spec
from test_cifar100_pipeline import FakeFederatedDataset, datasets


class SpecTests(unittest.TestCase):
    def test_cifar100_has_five_disjoint_tasks_of_twenty_classes(self):
        tasks = get_spec("cifar100").tasks()
        self.assertEqual([len(t) for t in tasks], [20] * 5)
        self.assertEqual(sorted(c for t in tasks for c in t), list(range(100)))
        self.assertEqual(tasks[0], list(range(20)))

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            get_spec("mnist")


@unittest.skipIf(datasets is None, "requiere torch, datasets y flwr-datasets")
class TaskDataTests(unittest.TestCase):
    def run_experiment(self, *extra):
        import cfl_tareas

        cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            with mock.patch("flwr_datasets.FederatedDataset", FakeFederatedDataset):
                cfl_tareas.main(["--quick", "--output-dir", tmp, *extra])
        finally:
            os.chdir(cwd)
        return Path(tmp)

    def test_quick_loads_five_tasks_per_client(self):
        out = self.run_experiment()
        manifest = json.loads((out / "data_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tasks"], get_spec("cifar100").tasks())
        self.assertEqual(len(manifest["clients"]), 2)
        for client in manifest["clients"]:
            self.assertEqual([t["task"] for t in client["tasks"]], [1, 2, 3, 4, 5])
            self.assertTrue(all(0 < t["samples"] <= 100 for t in client["tasks"]))
        self.assertTrue(all(t["classes_present"] == 20 for t in manifest["test"]))

    def test_each_task_only_holds_its_own_classes(self):
        import cfl_tareas

        args = cfl_tareas.build_parser().parse_args(["--quick"])
        config, spec = cfl_tareas.resolve_config(args, cfl_tareas.build_parser())
        experiment = cfl_tareas.TaskExperiment(args, config, spec)
        cwd, tmp = os.getcwd(), tempfile.mkdtemp()
        os.chdir(tmp)
        try:
            with mock.patch("flwr_datasets.FederatedDataset", FakeFederatedDataset):
                experiment.load_data()
        finally:
            os.chdir(cwd)
        for t, classes in enumerate(spec.tasks()):
            self.assertTrue(set(experiment.test[t][1].tolist()) <= set(classes))
            for client in experiment.train.values():
                self.assertTrue(set(client[t][1].tolist()) <= set(classes))

    def test_memories_and_task_evaluation(self):
        import cfl_tareas
        import fl_core

        args = cfl_tareas.build_parser().parse_args(["--quick"])
        config, spec = cfl_tareas.resolve_config(args, cfl_tareas.build_parser())
        experiment = cfl_tareas.TaskExperiment(args, config, spec)
        cwd, tmp = os.getcwd(), tempfile.mkdtemp()
        os.chdir(tmp)
        try:
            with mock.patch("flwr_datasets.FederatedDataset", FakeFederatedDataset):
                experiment.load_data()
        finally:
            os.chdir(cwd)
        memories = experiment.new_memories(20)
        for client, memory in memories.items():
            samples = sum(len(y) for _, y in experiment.train[client])
            self.assertEqual(memory.capacity, 20 * samples // 500)
            memory.add_task(*experiment.train[client][0])
            self.assertTrue(set(memory.tensors()[1].tolist()) <= set(spec.tasks()[0]))
        result = experiment.evaluate_tasks(fl_core.get_model(), 2)
        self.assertEqual(len(result["per_task"]), 3)
        self.assertTrue(0 <= result["pred_new_on_old"] <= 100)
        self.assertIsNone(experiment.evaluate_tasks(fl_core.get_model(), 0)["pred_new_on_old"])

    def test_config_records_fixed_hyperparameters(self):
        import cfl_tareas

        out = self.run_experiment("--metodo", "balanced", "--seed", "43")
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["fijos"], cfl_tareas.FIJOS)
        self.assertEqual((config["metodo"], config["seed"], config["num_tasks"]), ("balanced", 43, 5))
        self.assertEqual(config["fijos"]["local_epochs"], 5)

    def test_dirichlet_partition_loads(self):
        out = self.run_experiment("--partition", "dirichlet")
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["dirichlet_alpha"], 0.5)


class ParserTests(unittest.TestCase):
    def parse(self, *argv):
        import cfl_tareas

        parser = cfl_tareas.build_parser()
        return cfl_tareas.resolve_config(parser.parse_args(list(argv)), parser)

    def test_buffers_must_include_zero(self):
        with self.assertRaises(SystemExit):
            self.parse("--buffers", "10", "20")

    def test_hyperparameters_are_not_exposed(self):
        with self.assertRaises(SystemExit):
            self.parse("--local-epochs", "10")

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--metodo", "lwf")


if __name__ == "__main__":
    unittest.main()
