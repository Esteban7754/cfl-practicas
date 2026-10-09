"""Pruebas del orquestador de planes y del informe (sin entrenar: tareas simuladas)."""

import argparse
import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import informe_plan
import plan_experimentos

ROOT = Path(__file__).resolve().parent
(ROOT / "resultados").mkdir(exist_ok=True)


def make_plan(out):
    return {"nombre": "prueba", "salida": str(out), "paralelo": 2, "comun": ["--quick"],
            "fase1": {"iid": [], "dir": ["--partition", "dirichlet"]},
            "trabajos": [{"id": "a", "fase1": "iid", "semillas": [42, 43], "args": ["--buffers", "0", "20"]},
                         {"id": "b", "fase1": "iid", "semillas": [42], "args": ["--replay-mix", "balanced"]},
                         {"id": "c", "fase1": "dir", "semillas": [42], "args": []}]}


class BuildTests(unittest.TestCase):
    def test_phase1_is_trained_once_per_seed_and_partition(self):
        tasks = plan_experimentos.build_tasks(make_plan("resultados/x"))
        phase1 = [t for t in tasks if t["id"].startswith("fase1_")]
        self.assertEqual(sorted(t["id"] for t in phase1), ["fase1_dir_seed42", "fase1_iid_seed42", "fase1_iid_seed43"])
        self.assertTrue(all("--only-phase1" in t["args"] for t in phase1))
        job = next(t for t in tasks if t["id"] == "b_seed42")
        self.assertEqual(job["deps"], ["fase1_iid_seed42"])
        checkpoint = job["args"][job["args"].index("--phase1-checkpoint") + 1]
        self.assertTrue(checkpoint.endswith("phase1_checkpoint.pt") and "iid_seed42" in checkpoint)
        self.assertIn("--partition", next(t for t in tasks if t["id"] == "c_seed42")["args"])


class RunTests(unittest.TestCase):
    def run_with(self, script, out):
        """Ejecuta el plan con un sustituto del experimento: ``script`` decide qué hace cada tarea."""
        plan = make_plan(out)
        fake = lambda task: [sys.executable, "-c", script, str(ROOT / task["dir"]), task["done"], task["id"]]
        args = argparse.Namespace(dry_run=False, paralelo=None, informe_cada=60)
        with mock.patch.object(plan_experimentos, "command", fake), mock.patch.object(plan_experimentos.time, "sleep"):
            code = plan_experimentos.run_plan(plan, args)
        return code, json.loads((ROOT / out / "estado.json").read_text())

    def test_runs_everything_resumes_and_skips_dependents_of_failures(self):
        ok = "import sys, pathlib; p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); (p / sys.argv[2]).write_text('x')"
        failing_dir = ("import sys, pathlib; p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True)\n"
                       "sys.exit(1) if sys.argv[3] == 'fase1_dir_seed42' else (p / sys.argv[2]).write_text('x')")
        with tempfile.TemporaryDirectory(dir=ROOT / "resultados") as tmp:
            out = Path(tmp).relative_to(ROOT)
            code, status = self.run_with(failing_dir, out)
            self.assertEqual(code, 1)
            self.assertEqual(status["fase1_dir_seed42"], "fallida")
            self.assertEqual(status["c_seed42"], "omitida")
            self.assertEqual(status["a_seed43"], "hecha")
            # Al reanudar solo se repite lo que falta; la carpeta a medias se aparta.
            code, status = self.run_with(ok, out)
            self.assertEqual(code, 0)
            self.assertTrue(all(s == "hecha" for s in status.values()))
            self.assertTrue(list((ROOT / out / "_fase1").glob("dir_seed42_incompleta_*")))


class DeadlineTests(unittest.TestCase):
    def test_clock_time_in_the_past_means_tomorrow(self):
        now = datetime(2026, 10, 12, 18, 0)
        self.assertEqual(datetime.fromtimestamp(plan_experimentos.deadline_from("23:30", now=now)), datetime(2026, 10, 12, 23, 30))
        self.assertEqual(datetime.fromtimestamp(plan_experimentos.deadline_from("02:00", now=now)), datetime(2026, 10, 13, 2, 0))
        both = plan_experimentos.deadline_from("23:30", horas=1, now=now)  # manda el límite más cercano
        self.assertEqual(datetime.fromtimestamp(both), datetime(2026, 10, 12, 19, 0))
        self.assertIsNone(plan_experimentos.deadline_from())

    def test_pauses_without_losing_work_and_resumes(self):
        ok = "import sys, pathlib; p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); (p / sys.argv[2]).write_text('x')"
        plan = make_plan(None)
        fake = lambda task: [sys.executable, "-c", ok, str(ROOT / task["dir"]), task["done"], task["id"]]
        with tempfile.TemporaryDirectory(dir=ROOT / "resultados") as tmp, \
                mock.patch.object(plan_experimentos, "command", fake), mock.patch.object(plan_experimentos.time, "sleep"):
            plan["salida"] = str(Path(tmp).relative_to(ROOT))
            paused = argparse.Namespace(dry_run=False, paralelo=None, informe_cada=60, hasta=None, horas=0)
            self.assertEqual(plan_experimentos.run_plan(plan, paused), plan_experimentos.PAUSED)
            status = json.loads((Path(tmp) / "estado.json").read_text())
            self.assertTrue(all(s == "pendiente" for s in status.values()))  # no empezó nada
            resumed = argparse.Namespace(dry_run=False, paralelo=None, informe_cada=60, hasta=None, horas=None)
            self.assertEqual(plan_experimentos.run_plan(plan, resumed), 0)


class GpuTests(unittest.TestCase):
    def test_tasks_are_spread_over_the_gpus(self):
        record = ("import os, sys, pathlib; p = pathlib.Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True); "
                  "(p / sys.argv[2]).write_text(os.environ.get('CUDA_VISIBLE_DEVICES', 'ninguna'))")
        with tempfile.TemporaryDirectory(dir=ROOT / "resultados") as tmp, \
                mock.patch.object(plan_experimentos, "gpu_count", return_value=2), \
                mock.patch.dict(plan_experimentos.os.environ, {}, clear=False) as env:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            out = Path(tmp).relative_to(ROOT)
            code, _ = RunTests.run_with(self, record, out)
            self.assertEqual(code, 0)
            used = {p.read_text() for p in (ROOT / out).rglob("*") if p.name in ("results.csv", "phase1_checkpoint.pt")}
            self.assertEqual(used, {"0", "1"})


class ReportTests(unittest.TestCase):
    def test_report_handles_partial_results(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "resultados") as tmp:
            out = Path(tmp)
            plan = {"nombre": "prueba", "salida": str(out.relative_to(ROOT)),
                    "trabajos": [{"id": "principal_normal", "semillas": [42, 43]},
                                 {"id": "principal_equilibrados", "semillas": [42, 43]},
                                 {"id": "fraccion_025", "semillas": [42]}, {"id": "fraccion_075", "semillas": [42]}]}
            for job, a in (("principal_normal", 20.0), ("principal_equilibrados", 40.0), ("fraccion_025", 30.0)):
                for seed in (42, 43):
                    folder = out / job / f"seed_{seed}"
                    folder.mkdir(parents=True)
                    with open(folder / "results.csv", "w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["fraction", "step4_acc_a", "retention_a_percent",
                                                                    "acc_all_100_classes", "wa_acc_all_100_classes"])
                        writer.writeheader()
                        for fraction in ("No Replay", "10% Replay", "20% Replay"):
                            writer.writerow({"fraction": fraction, "step4_acc_a": a + seed - 42, "retention_a_percent": a,
                                             "acc_all_100_classes": a / 2, "wa_acc_all_100_classes": a / 2 + 1})
            plan_path = out / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            informe_plan.main(plan_path)
            report = (out / "INFORME.md").read_text(encoding="utf-8")
            self.assertIn("20,5 ± 0,7", report)          # media ± desviación de las dos semillas
            self.assertIn("sin resultados todavía", report)  # fraccion_075 aún no ha corrido
            self.assertIn("Ablación de la fracción", report)
            self.assertTrue((out / "informe" / "fraccion.png").is_file())


if __name__ == "__main__":
    unittest.main()
