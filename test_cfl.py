"""Pruebas de la interfaz cfl.py (sin entrenar: modo --dry-run)."""

import contextlib
import io
import unittest
from unittest import mock

import cfl


def commands_of(argv):
    """Ejecuta cfl en modo simulación y devuelve los comandos que lanzaría."""
    calls = []

    def fake_run(command, log=None, env=None, dry_run=False):
        calls.append(([str(c) for c in command], str(log) if log else None))
        return 0

    with mock.patch.object(cfl, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
        cfl.main(argv)
    return calls


class CompararTests(unittest.TestCase):
    def test_one_seed_runs_experiment_and_verification_in_same_folder(self):
        calls = commands_of(["comparar", "--output-dir", "resultados/x_test", "--dry-run"])
        experiment, verification = calls
        self.assertIn("cifar100_three_buffer.py", experiment[0])
        self.assertIn("--controlled-replay", experiment[0])
        self.assertEqual(experiment[0][experiment[0].index("--output-dir") + 1], "resultados/x_test")
        self.assertIn("verificar_cifar100.py", verification[0])
        self.assertTrue(verification[1].endswith("verificacion.log"))

    def test_several_seeds_use_subfolders_and_summary(self):
        calls = commands_of(["comparar", "--seeds", "42", "43", "--buffers", "0", "10", "--replay-mix", "balanced",
                             "--distill-weight", "1", "--output-dir", "y_test", "--dry-run"])
        dirs = [c[0][c[0].index("--output-dir") + 1] for c in calls if "cifar100_three_buffer.py" in c[0]]
        self.assertEqual(dirs, ["resultados/y_test/seed_42", "resultados/y_test/seed_43"])
        self.assertIn("agregar_semillas.py", calls[-1][0])
        self.assertIn("--replay-batch-fraction", calls[0][0])

    def test_conjunto_adds_the_joint_baseline_only_when_asked(self):
        with_joint = commands_of(["comparar", "--conjunto", "--output-dir", "z_test", "--dry-run"])[0][0]
        without = commands_of(["comparar", "--output-dir", "z_test", "--dry-run"])[0][0]
        self.assertIn("--joint", with_joint)
        self.assertNotIn("--joint", without)

    def test_plan_defaults_to_the_gpu_plan(self):
        args = cfl.build_parser().parse_args(["plan", "--paralelo", "2"])
        self.assertEqual((args.plan_file, args.paralelo, args.func), ("planes/plan_gpu.json", 2, cfl.cmd_plan))

    def test_rejects_missing_no_replay_and_incoherent_checkpoint(self):
        with self.assertRaises(SystemExit):
            commands_of(["comparar", "--buffers", "10", "20", "--dry-run"])
        with self.assertRaises(SystemExit):
            commands_of(["comparar", "--reuse-checkpoint", "--local-epochs", "5", "--dry-run"])

    def test_output_dir_must_stay_inside_project(self):
        for bad in ("/tmp/fuera", "../fuera", "resultados/../../fuera"):
            with self.assertRaises(SystemExit):
                cfl.new_output_dir(bad, "x")


class SemillasTests(unittest.TestCase):
    def test_unknown_arguments_go_to_the_script(self):
        calls = commands_of(["semillas", "domainnet.py", "--seeds", "1", "2", "--buffers", "0", "10",
                             "--output-dir", "z_test", "--dry-run"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0][1:], ["domainnet.py", "--seed", "1", "--output-dir", "resultados/z_test/seed_1",
                                           "--buffers", "0", "10"])
        self.assertIn("agregar_semillas.py", calls[-1][0])

    def test_other_commands_reject_unknown_arguments(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            commands_of(["rapido", "--desconocido"])


class InfoTests(unittest.TestCase):
    def test_info_prints_environment(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cfl.main(["info"]), 0)
        self.assertIn("packages_sha256", out.getvalue())


if __name__ == "__main__":
    unittest.main()
