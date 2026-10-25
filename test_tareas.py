"""Tests de cfl_tareas.py (protocolo de varias tareas) con datos sintéticos y sin red.

Reutiliza el ``FakeFederatedDataset`` de test_cifar100_pipeline.py.
"""

import csv
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
    def run_experiment(self, *extra, only_data=False):
        """main() con datos sintéticos; con ``only_data`` solo carga los datos (sin entrenar)."""
        import cfl_tareas

        cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        run = (lambda self: self.load_data()) if only_data else cfl_tareas.TaskExperiment.run
        try:
            with mock.patch("flwr_datasets.FederatedDataset", FakeFederatedDataset), \
                    mock.patch.object(cfl_tareas.TaskExperiment, "run", run):
                cfl_tareas.main(["--quick", "--output-dir", tmp, *extra])
        finally:
            os.chdir(cwd)
        return Path(tmp)

    def loaded_experiment(self, *argv):
        import cfl_tareas

        args = cfl_tareas.build_parser().parse_args(["--quick", *argv])
        config, spec = cfl_tareas.resolve_config(args, cfl_tareas.build_parser())
        experiment = cfl_tareas.TaskExperiment(args, config, spec)
        cwd, tmp = os.getcwd(), tempfile.mkdtemp()
        os.chdir(tmp)
        try:
            with mock.patch("flwr_datasets.FederatedDataset", FakeFederatedDataset):
                experiment.load_data()
        finally:
            os.chdir(cwd)
        return experiment, spec

    def test_quick_loads_five_tasks_per_client(self):
        out = self.run_experiment(only_data=True)
        manifest = json.loads((out / "data_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tasks"], get_spec("cifar100").tasks())
        self.assertEqual(len(manifest["clients"]), 2)
        for client in manifest["clients"]:
            self.assertEqual([t["task"] for t in client["tasks"]], [1, 2, 3, 4, 5])
            self.assertTrue(all(0 < t["samples"] <= 100 for t in client["tasks"]))
        self.assertTrue(all(t["classes_present"] == 20 for t in manifest["test"]))

    def test_each_task_only_holds_its_own_classes(self):
        experiment, spec = self.loaded_experiment()
        for t, classes in enumerate(spec.tasks()):
            self.assertTrue(set(experiment.test[t][1].tolist()) <= set(classes))
            for client in experiment.train.values():
                self.assertTrue(set(client[t][1].tolist()) <= set(classes))

    def test_memories_and_task_evaluation(self):
        import fl_core

        experiment, spec = self.loaded_experiment()
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

    def read_csv(self, path):
        with open(path, encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_every_method_runs_all_tasks_and_writes_metrics(self):
        no_replay = {}
        for method in ("concat", "balanced", "calendario", "ace"):
            with self.subTest(method=method):
                out = self.run_experiment("--metodo", method, "--buffers", "0", "20", "--audit",
                                          *(["--joint"] if method == "balanced" else []))
                results = self.read_csv(out / "results.csv")
                self.assertEqual([r["buffer"] for r in results], ["0", "20"])
                self.assertEqual(results[0]["crt_acc_final"], "")  # cRT solo con memoria
                self.assertNotEqual(results[1]["crt_acc_final"], "")
                self.assertGreater(int(results[1]["memoria_total"]), 0)
                matrix = self.read_csv(out / "matriz.csv")
                self.assertEqual(len(matrix), 2 * (1 + 2 + 3 + 4 + 5))
                last = [r for r in matrix if r["buffer"] == "20" and r["tras_tarea"] == "5"]
                self.assertEqual(float(last[0]["acc"]), float(results[1]["acc_t1_final"]))
                self.assertEqual(len(self.read_csv(out / "history.csv")), 2 * 4)
                no_replay[method] = results[0]["acc_final"]
                if method == "balanced":
                    joint = self.read_csv(out / "joint_results.csv")[0]
                    self.assertEqual(joint["rounds"], "5")
                    self.assertIn("gap_vs_joint", results[0])
                audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
                steps = {(r["context"]["buffer"], r["context"]["task"], r["context"]["client"]): r["optimizer_steps"]
                         for r in audit["training"] if "buffer" in r["context"]}
                for (buffer, task, client), n in steps.items():  # mismo presupuesto con y sin memoria
                    self.assertEqual(n, steps[(0, task, client)])
        self.assertEqual(len(set(no_replay.values())), 1)  # sin memoria, el método no cambia nada

    def test_reusing_the_first_task_gives_the_same_results(self):
        phase1 = self.run_experiment("--only-phase1")
        checkpoint = phase1 / "phase1_checkpoint.pt"
        manifest = json.loads((phase1 / "phase1_checkpoint.config.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], 42)
        self.assertFalse((phase1 / "results.csv").exists())
        args = ("--metodo", "balanced", "--buffers", "0", "20")
        trained = self.read_csv(self.run_experiment(*args) / "results.csv")
        reused = self.read_csv(self.run_experiment(*args, "--phase1-checkpoint", str(checkpoint)) / "results.csv")
        self.assertEqual(trained, reused)

    def test_checkpoint_from_another_setup_is_rejected(self):
        phase1 = self.run_experiment("--only-phase1")
        checkpoint = phase1 / "phase1_checkpoint.pt"
        with self.assertRaises(SystemExit):  # otra semilla
            self.run_experiment("--seed", "43", "--phase1-checkpoint", str(checkpoint), only_data=True)
        with self.assertRaises(SystemExit):  # otro reparto
            self.run_experiment("--partition", "dirichlet", "--phase1-checkpoint", str(checkpoint), only_data=True)
        manifest_path = phase1 / "phase1_checkpoint.config.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.write_text(json.dumps({**manifest, "checkpoint_sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaises(SystemExit):  # pesos que no son los del manifiesto
            self.run_experiment("--phase1-checkpoint", str(checkpoint), only_data=True)
        manifest_path.unlink()
        with self.assertRaises(SystemExit):  # sin manifiesto
            self.run_experiment("--phase1-checkpoint", str(checkpoint), only_data=True)

    def test_config_records_fixed_hyperparameters(self):
        import cfl_tareas

        out = self.run_experiment("--metodo", "balanced", "--seed", "43", only_data=True)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["fijos"], cfl_tareas.FIJOS)
        self.assertEqual((config["metodo"], config["seed"], config["num_tasks"]), ("balanced", 43, 5))
        self.assertEqual(config["fijos"]["local_epochs"], 5)

    def test_dirichlet_partition_runs_end_to_end(self):
        out = self.run_experiment("--partition", "dirichlet", "--metodo", "ace", "--buffers", "0", "20")
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["dirichlet_alpha"], 0.5)
        self.assertEqual(len(self.read_csv(out / "results.csv")), 2)


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
