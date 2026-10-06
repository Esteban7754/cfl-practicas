"""Pruebas de las utilidades comunes y del resumen por semillas (no necesitan PyTorch)."""

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment_common import build_parser, apply_args, make_result, nested_buffer_indices, save_results_csv, step_budget

HERE = Path(__file__).resolve().parent


class BufferTests(unittest.TestCase):
    def test_buffers_are_nested_and_sized_like_before(self):
        for client in range(3):
            small = set(nested_buffer_indices(250, 0.05, 42, client))
            large = set(nested_buffer_indices(250, 0.20, 42, client))
            self.assertEqual(len(small), int(250 * 0.05))
            self.assertEqual(len(large), int(250 * 0.20))
            self.assertTrue(small <= large)

    def test_zero_percent_is_empty_and_global_rng_untouched(self):
        np.random.seed(1)
        expected = np.random.random(2)
        np.random.seed(1)
        self.assertEqual(nested_buffer_indices(100, 0.0, 42, 0), [])
        nested_buffer_indices(100, 0.25, 42, 0)
        np.testing.assert_array_equal(np.random.random(2), expected)

    def test_seed_and_client_change_the_selection(self):
        base = nested_buffer_indices(500, 0.1, 42, 0)
        self.assertEqual(base, nested_buffer_indices(500, 0.1, 42, 0))
        self.assertNotEqual(base, nested_buffer_indices(500, 0.1, 43, 0))
        self.assertNotEqual(base, nested_buffer_indices(500, 0.1, 42, 1))

    def test_step_budget(self):
        self.assertIsNone(step_budget(2500, 32, 5, equal_steps=False))
        self.assertEqual(step_budget(2500, 32, 5, equal_steps=True), 79 * 5)


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_no_replay_must_be_measured(self):
        parser = build_parser("x")
        with self.assertRaises(SystemExit):
            apply_args(parser.parse_args(["--buffers", "5", "15"]), {"seed": 42}, parser)

    def test_overrides_and_default_output_dir(self):
        parser = build_parser("x")
        config = {"seed": 42, "num_rounds": 5, "local_epochs": 5}
        apply_args(parser.parse_args(["--seed", "7", "--local-epochs", "2", "--buffers", "20", "0", "--equal-steps"]), config, parser)
        self.assertEqual((config["seed"], config["local_epochs"], config["buffers"], config["equal_steps"]), (7, 2, [0, 20], True))
        self.assertIn(os.path.join("resultados", ""), os.getcwd() + os.sep)
        self.assertTrue(os.path.exists("config.json"))


class AggregationTests(unittest.TestCase):
    def test_mean_and_std_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, loss in ((42, 10.0), (43, 14.0)):
                folder = Path(tmp, f"seed_{seed}")
                folder.mkdir()
                save_results_csv([make_result(0, 50.0, 50.0 - 40.0, 30.0), make_result(20, 50.0, 50.0 - loss, 30.0)],
                                 folder / "results.csv")
            subprocess.run([sys.executable, str(HERE / "agregar_semillas.py"), tmp], check=True, capture_output=True,
                           env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(HERE)})
            with open(Path(tmp, "resumen_semillas.csv"), encoding="utf-8") as handle:
                rows = {int(r["replay_percent"]): r for r in csv.DictReader(handle)}
            self.assertAlmostEqual(float(rows[20]["loss_a_pp_mean"]), 12.0)
            self.assertAlmostEqual(float(rows[20]["loss_a_pp_std"]), 2 ** 0.5 * 2)
            self.assertEqual(int(rows[0]["n_seeds"]), 2)
            self.assertTrue(Path(tmp, "perdida_A_media_semillas.png").exists())


if __name__ == "__main__":
    unittest.main()
