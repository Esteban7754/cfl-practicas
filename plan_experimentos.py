"""Ejecuta un plan de experimentos (JSON) de principio a fin, sin intervención.

    python plan_experimentos.py planes/plan_gpu.json             # ejecuta lo que falte
    python plan_experimentos.py planes/plan_gpu.json --dry-run   # lista las tareas y los comandos
    python plan_experimentos.py planes/plan_gpu.json --solo-informe

- Primero entrena la fase 1 de cada semilla y reparto una sola vez (``--only-phase1``);
  todos los trabajos de esa semilla y reparto la reutilizan con ``--phase1-checkpoint``.
- Lanza hasta ``paralelo`` tareas a la vez en la misma GPU, por orden de prioridad (el del plan).
- Es reanudable: una tarea con su results.csv (o su phase1_checkpoint.pt) ya está hecha y se
  salta; una carpeta a medias se aparta como ``*_incompleta_<fecha>`` y se repite.
- Un solo Ctrl+C (o la parada del contenedor) detiene todas las tareas en curso.
- ``--hasta HH:MM`` o ``--horas N``: a partir de ese momento no empieza tareas nuevas, deja
  terminar las que están en curso, escribe el informe parcial y se para (código 3). Al relanzar,
  sigue con lo pendiente. Así el plan se reparte en varias tandas sin perder nada.
- Al terminar agrega las semillas de cada trabajo (agregar_semillas.py) y escribe el informe
  (informe_plan.py): resultados/<plan>/INFORME.md con tablas y gráficas.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_tasks(plan):
    out = Path(plan["salida"])
    common = plan.get("comun", [])
    phase1, tasks = {}, []
    for job in plan["trabajos"]:
        key = job.get("fase1", "iid")
        if key not in plan["fase1"]:
            raise SystemExit(f"El trabajo {job['id']} usa la fase 1 '{key}', que no está en el plan")
        for seed in job["semillas"]:
            p1 = phase1.setdefault((key, seed), {
                "id": f"fase1_{key}_seed{seed}", "dir": out / "_fase1" / f"{key}_seed{seed}", "deps": [],
                "done": "phase1_checkpoint.pt",
                "args": [*common, *plan["fase1"][key], "--seed", str(seed), "--only-phase1", "--buffers", "0"]})
            tasks.append({
                "id": f"{job['id']}_seed{seed}", "job": job["id"], "dir": out / job["id"] / f"seed_{seed}",
                "deps": [p1["id"]], "done": "results.csv",
                "args": [*common, *plan["fase1"][key], *job["args"], "--seed", str(seed),
                         "--phase1-checkpoint", str(p1["dir"] / "phase1_checkpoint.pt")]})
    return list(phase1.values()) + tasks


def is_done(task):
    return (ROOT / task["dir"] / task["done"]).is_file()


def command(task):
    return [sys.executable, "cifar100_three_buffer.py", *task["args"], "--output-dir", str(task["dir"])]


PAUSED = 3  # código de salida cuando se para por el límite de tiempo con tareas pendientes


def deadline_from(hasta=None, horas=None, now=None):
    """Momento (epoch) a partir del cual no se empiezan tareas nuevas; None = sin límite.

    ``hasta`` = "HH:MM" del reloj local; si esa hora ya pasó hoy, es la de mañana.
    """
    now = now or datetime.now()
    limits = []
    if horas is not None:
        limits.append(now.timestamp() + horas * 3600)
    if hasta:
        hour, minute = (int(v) for v in hasta.split(":"))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        limits.append(target.timestamp())
    return min(limits) if limits else None


def parse_clock(value):
    try:
        hour, minute = (int(v) for v in value.split(":"))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("usa el formato HH:MM, por ejemplo 23:30")


def gpu_count():
    """GPUs visibles (0 sin GPU). Con varias, las tareas se reparten entre ellas."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # sin torch o sin CUDA: se ejecuta todo en CPU
        return 0


def last_progress(log):
    try:
        lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if any(k in l for k in ("Ronda", "FASE", "REFERENCIA", "Traceback", "Error"))]
        return lines[-1].strip()[:110] if lines else "(arrancando)"
    except OSError:
        return "(sin registro)"


def run_plan(plan, args):
    out = ROOT / plan["salida"]
    tasks = build_tasks(plan)
    if args.dry_run:
        for t in tasks:
            state = "hecha" if is_done(t) else "pendiente"
            print(f"[{state:9s}] {t['id']}\n    {' '.join(command(t))}")
        print(f"\n{sum(not is_done(t) for t in tasks)} tareas pendientes de {len(tasks)}")
        return 0

    logs = out / "_registros"
    logs.mkdir(parents=True, exist_ok=True)
    parallel = args.paralelo or plan.get("paralelo", 1)
    threads = max(1, (os.cpu_count() or 2) // parallel)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    pending = [t for t in tasks if not is_done(t)]
    status = {t["id"]: ("hecha" if is_done(t) else "pendiente") for t in tasks}
    running = {}
    start = time.time()
    gpus = gpu_count() if "CUDA_VISIBLE_DEVICES" not in os.environ else 0
    deadline = deadline_from(getattr(args, "hasta", None), getattr(args, "horas", None))
    print(f"[PLAN] {plan['nombre']}: {len(pending)} tareas pendientes de {len(tasks)}; {parallel} en paralelo, "
          f"{threads} hilos de CPU cada una{f', repartidas entre {gpus} GPU' if gpus > 1 else ''}. "
          f"Resultados en {plan['salida']}", flush=True)
    if deadline is not None:
        print(f"[PLAN] No se empezarán tareas nuevas a partir de las "
              f"{datetime.fromtimestamp(deadline):%H:%M del %d/%m}; las que estén en curso terminarán.", flush=True)

    def stop_all(*_):
        print("\n[PLAN] Deteniendo todas las tareas en curso...", flush=True)
        for proc, _task, handle in running.values():
            proc.terminate()
        for proc, _task, handle in running.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            handle.close()
        print("[PLAN] Detenido. Vuelve a lanzar el mismo comando para continuar donde se quedó.", flush=True)
        raise SystemExit(130)

    previous = signal.signal(signal.SIGINT, stop_all), signal.signal(signal.SIGTERM, stop_all)
    try:
        return schedule(pending, running, status, tasks, parallel, env, logs, out, start, args, gpus, deadline)
    finally:
        signal.signal(signal.SIGINT, previous[0])
        signal.signal(signal.SIGTERM, previous[1])


def schedule(pending, running, status, tasks, parallel, env, logs, out, start, args, gpus=0, deadline=None):
    next_report, announced = 0.0, False
    while pending or running:
        if deadline is not None and time.time() >= deadline and pending:
            if not running:
                (out / "estado.json").write_text(json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"\n[PLAN] Pausa: alcanzado el límite de tiempo tras {(time.time() - start) / 3600:.2f} h. "
                      f"Hechas {sum(s == 'hecha' for s in status.values())} de {len(tasks)}; quedan {len(pending)}. "
                      "Vuelve a lanzar el mismo comando para continuar.", flush=True)
                return PAUSED
            if not announced:
                print(f"[PLAN] Límite de tiempo alcanzado: no se empiezan tareas nuevas; esperando a las "
                      f"{len(running)} en curso.", flush=True)
                announced = True
            pending_view = []  # no se lanza nada; solo se recogen las que terminan
        else:
            pending_view = pending
        for tid, (proc, task, handle) in list(running.items()):
            if proc.poll() is not None:
                handle.close()
                del running[tid]
                ok = proc.returncode == 0 and is_done(task)
                status[tid] = "hecha" if ok else "fallida"
                print(f"[PLAN] {'OK    ' if ok else 'FALLO '} {tid} ({(time.time() - start) / 3600:.2f} h)"
                      + ("" if ok else f" -> {logs / (tid + '.log')}"), flush=True)
        for task in list(pending_view):
            if len(running) >= parallel:
                break
            deps = [status[d] for d in task["deps"]]
            if any(d == "fallida" or d == "omitida" for d in deps):
                status[task["id"]] = "omitida"
                pending.remove(task)
                print(f"[PLAN] OMITIDA {task['id']} (falló su fase 1)", flush=True)
                continue
            if any(d != "hecha" for d in deps):
                continue
            folder = ROOT / task["dir"]
            if folder.exists():
                folder.rename(folder.with_name(f"{folder.name}_incompleta_{datetime.now():%Y%m%d_%H%M%S}"))
            folder.parent.mkdir(parents=True, exist_ok=True)
            handle = open(logs / f"{task['id']}.log", "w", encoding="utf-8")
            task_env, where = env, ""
            if gpus > 1:  # a la GPU con menos tareas en curso
                load = [sum(t.get("gpu") == g for _p, t, _h in running.values()) for g in range(gpus)]
                task["gpu"] = load.index(min(load))
                task_env, where = {**env, "CUDA_VISIBLE_DEVICES": str(task["gpu"])}, f" (GPU {task['gpu']})"
            proc = subprocess.Popen(command(task), cwd=ROOT, env=task_env, stdout=handle, stderr=subprocess.STDOUT)
            running[task["id"]] = (proc, task, handle)
            status[task["id"]] = "en curso"
            pending.remove(task)
            print(f"[PLAN] INICIO {task['id']}{where}", flush=True)
        if time.time() >= next_report and running:
            done = sum(s == "hecha" for s in status.values())
            print(f"--- {(time.time() - start) / 3600:.2f} h | hechas {done}/{len(tasks)} | "
                  f"fallidas {sum(s == 'fallida' for s in status.values())}", flush=True)
            for tid, (_p, _t, _h) in running.items():
                print(f"    {tid:34s} {last_progress(logs / (tid + '.log'))}", flush=True)
            next_report = time.time() + args.informe_cada * 60
        (out / "estado.json").write_text(json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8")
        time.sleep(5)

    failed = [t for t, s in status.items() if s in ("fallida", "omitida")]
    print(f"\n[PLAN] Terminado en {(time.time() - start) / 3600:.2f} h. "
          f"{'Todo correcto.' if not failed else 'Con problemas: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


def summarize(plan, plan_path):
    out = ROOT / plan["salida"]
    for job in plan["trabajos"]:
        folder = out / job["id"]
        if len(list(folder.glob("seed_*/results.csv"))) >= 2:
            with open(folder / "resumen_semillas.log", "w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "agregar_semillas.py", str(folder)], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
    subprocess.run([sys.executable, "informe_plan.py", str(plan_path)], cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--solo-informe", action="store_true", help="Solo agrega e informa con lo que ya haya")
    parser.add_argument("--paralelo", type=int, help="Tareas simultáneas (por defecto, las del plan)")
    parser.add_argument("--informe-cada", type=float, default=10, help="Minutos entre informes de progreso")
    parser.add_argument("--hasta", type=parse_clock, help="Hora (HH:MM) a partir de la cual no se empiezan tareas nuevas")
    parser.add_argument("--horas", type=float, help="Horas a partir de las cuales no se empiezan tareas nuevas")
    args = parser.parse_args(argv)
    plan = json.loads((ROOT / args.plan).read_text(encoding="utf-8"))
    code = 0 if args.solo_informe else run_plan(plan, args)
    if not args.dry_run:
        summarize(plan, ROOT / args.plan)
    return code


if __name__ == "__main__":
    sys.exit(main())
