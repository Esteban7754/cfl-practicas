"""Interfaz única de los experimentos (pensada para ejecutarse dentro de Docker).

    cfl info                      entorno, datos disponibles y versión de la imagen
    cfl entorno                   verificación rápida del entorno
    cfl tests                     tests unitarios
    cfl rapido                    prueba funcional con CIFAR-100 real + verificación
    cfl comparar [opciones]       comparación de replay completa y verificada
    cfl semillas SCRIPT [...]     repite un experimento con varias semillas y resume

Cualquier otro comando se ejecuta tal cual: «python domainnet.py --buffers 0 20».
Los resultados se guardan siempre en resultados/<carpeta nueva>.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
CHECKPOINT = ROOT / "global_model_phase1.pt"
SEED_SCRIPTS = ("cifar100_three_buffer.py", "domainnet.py", "mvtecad_all_experiments.py", "legacy/three_buffer_v2.py")


def stamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def new_output_dir(requested, default_name):
    """Carpeta relativa dentro de resultados/ que todavía no existe."""
    relative = Path(requested) if requested else Path("resultados") / f"{default_name}_{stamp()}"
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("La carpeta de salida debe ser una ruta relativa dentro del proyecto")
    if relative.parts[0] != "resultados":
        relative = Path("resultados") / relative
    target = ROOT / relative
    if target.exists():
        raise SystemExit(f"{relative} ya existe. Elige otra carpeta para conservar sus resultados.")
    return relative


def child_env(threads=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if threads:
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    return env


def run(command, log=None, env=None, dry_run=False):
    """Ejecuta un comando desde la raíz del proyecto; con ``log`` copia la salida a un archivo."""
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return 0
    if log is None:
        return subprocess.run(command, cwd=ROOT, env=env).returncode
    log = ROOT / log
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as handle, subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
    return process.returncode


def check(code, message):
    if code != 0:
        raise SystemExit(f"[ERROR] {message} (código {code})")


# ------------------------------------------------------------------
# Comandos
# ------------------------------------------------------------------
def cmd_info(args):
    import torch

    from reproducibility import environment_info

    info = environment_info()
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    info["checkpoint_global_model_phase1"] = CHECKPOINT.is_file()
    info["mvtec_disponible"] = (ROOT / "data" / "mvtec_anomaly_detection").is_dir()
    info["hf_home"] = os.environ.get("HF_HOME")
    info["resultados_escribible"] = os.access(RESULTS if RESULTS.exists() else ROOT, os.W_OK)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not info["cuda_available"] and info.get("image_variant") == "cuda":
        print("[AVISO] Imagen GPU sin GPU visible: ¿falta «--gpus all» o el controlador NVIDIA?")
    return 0


def cmd_entorno(args):
    return run([sys.executable, "verificar_entorno.py"], env=child_env())


def cmd_tests(args):
    return run([sys.executable, "-m", "unittest", *(args.extra or ["-v"])], env=child_env())


def cmd_rapido(args):
    out = new_output_dir(args.output_dir, "rapido")
    env = child_env(args.threads)
    check(run([sys.executable, "cifar100_three_buffer.py", "--quick", "--audit", "--output-dir", str(out)],
              log=out / "ejecucion.log", env=env, dry_run=args.dry_run), "La prueba rápida falló")
    code = verify(out, env, args.dry_run)
    print(f"\nPrueba rápida terminada en {out}. Recuerda: --quick comprueba el flujo, no el aprendizaje.")
    return code


def verify(out, env, dry_run):
    """Verifica desde los pesos; solo es exacto si se entrenó en CPU."""
    config_path = ROOT / out / "config.json"
    if not dry_run and config_path.is_file():
        device = json.loads(config_path.read_text(encoding="utf-8")).get("device")
        if device != "cpu":
            print(f"[AVISO] Entrenado en {device}: se omite la verificación exacta en CPU.")
            return 0
    code = run([sys.executable, "verificar_cifar100.py", str(out)], log=out / "verificacion.log", env=env, dry_run=dry_run)
    if code == 2:
        print("[AVISO] Métricas verificadas, pero la comparación es inconcluyente (ver verificacion.log).")
        return 0
    check(code, f"La verificación falló; consulta {out}/verificacion.log")
    return 0


def cmd_comparar(args):
    if 0 not in args.buffers:
        raise SystemExit("--buffers debe incluir 0: la referencia sin replay se mide")
    if args.reuse_checkpoint:
        if args.local_epochs != 1 or any(seed != 42 for seed in args.seeds):
            raise SystemExit("global_model_phase1.pt se entrenó con 1 época local y semilla 42: "
                             "--reuse-checkpoint solo es coherente con --local-epochs 1 --seeds 42")
        if not CHECKPOINT.is_file():
            raise SystemExit("No se encuentra global_model_phase1.pt (en Docker: monta la carpeta del proyecto con el modo dev)")
    variant = ("_balanced" if args.replay_mix == "balanced" else "") + ("_lwf" if args.distill_weight > 0 else "")
    out = new_output_dir(args.output_dir, f"cifar100_completo_{args.local_epochs}epocas{variant}")
    env = child_env(args.threads)
    for seed in args.seeds:
        run_dir = out / f"seed_{seed}" if len(args.seeds) > 1 else out
        command = [sys.executable, "cifar100_three_buffer.py", "--controlled-replay", "--audit", "--seed", str(seed),
                   "--buffers", *map(str, args.buffers), "--local-epochs", str(args.local_epochs),
                   "--rounds", str(args.rounds), "--replay-mix", args.replay_mix,
                   "--distill-weight", str(args.distill_weight), "--output-dir", str(run_dir)]
        if args.replay_mix == "balanced":
            command += ["--replay-batch-fraction", str(args.replay_batch_fraction)]
        if args.reuse_checkpoint:
            command += ["--phase1-checkpoint", str(CHECKPOINT), "--no-normalize", "--no-augment"]
        print(f"\n=== Semilla {seed}, {args.local_epochs} épocas locales -> {run_dir}")
        check(run(command, log=run_dir / "ejecucion.log", env=env, dry_run=args.dry_run),
              f"El experimento falló (semilla {seed}); consulta {run_dir}/ejecucion.log")
        verify(run_dir, env, args.dry_run)
    if len(args.seeds) > 1:
        check(run([sys.executable, "agregar_semillas.py", str(out)], log=out / "resumen_semillas.log", env=env,
                  dry_run=args.dry_run), "No se pudo generar el resumen de semillas")
    print(f"\nComparación terminada en {out}")
    return 0


def cmd_semillas(args):
    out = new_output_dir(args.output_dir, Path(args.script).stem + "_semillas")
    env = child_env(args.threads)
    for seed in args.seeds:
        seed_dir = out / f"seed_{seed}"
        check(run([sys.executable, args.script, "--seed", str(seed), "--output-dir", str(seed_dir), *args.extra],
                  log=seed_dir / "ejecucion.log", env=env, dry_run=args.dry_run),
              f"Falló la semilla {seed}; consulta {seed_dir}/ejecucion.log")
    check(run([sys.executable, "agregar_semillas.py", str(out)], log=out / "resumen_semillas.log", env=env,
              dry_run=args.dry_run), "No se pudo generar el resumen de semillas")
    print(f"\nTerminado. Resumen en {out}/resumen_semillas.csv")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="cfl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Entorno, datos y versión de la imagen").set_defaults(func=cmd_info)
    sub.add_parser("entorno", help="Verificación rápida del entorno").set_defaults(func=cmd_entorno)
    tests = sub.add_parser("tests", help="Tests unitarios (argumentos extra: los de unittest)")
    tests.add_argument("extra", nargs=argparse.REMAINDER)
    tests.set_defaults(func=cmd_tests)

    def common(p):
        p.add_argument("--output-dir", help="Carpeta dentro de resultados/ (por defecto, una nueva con fecha)")
        p.add_argument("--threads", type=int, help="Hilos de CPU (OMP/MKL); por defecto los que decida PyTorch")
        p.add_argument("--dry-run", action="store_true", help="Muestra los comandos sin ejecutarlos")

    rapido = sub.add_parser("rapido", help="Prueba funcional con CIFAR-100 real y verificación")
    common(rapido)
    rapido.set_defaults(func=cmd_rapido)

    comparar = sub.add_parser("comparar", help="Comparación de replay con todos los datos, verificada desde los pesos")
    common(comparar)
    comparar.add_argument("--seeds", nargs="+", type=int, default=[42])
    comparar.add_argument("--buffers", nargs="+", type=int, default=[0, 20])
    comparar.add_argument("--local-epochs", type=int, default=5)
    comparar.add_argument("--rounds", type=int, default=5)
    comparar.add_argument("--replay-mix", choices=["concat", "balanced"], default="concat")
    comparar.add_argument("--replay-batch-fraction", type=float, default=0.5)
    comparar.add_argument("--distill-weight", type=float, default=0.0)
    comparar.add_argument("--reuse-checkpoint", action="store_true",
                          help="Reutiliza global_model_phase1.pt (1 época, sin normalización ni aumento)")
    comparar.set_defaults(func=cmd_comparar)

    semillas = sub.add_parser("semillas", help="Repite un experimento con varias semillas y resume media ± desviación")
    common(semillas)
    semillas.add_argument("script", choices=SEED_SCRIPTS)
    semillas.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    semillas.set_defaults(func=cmd_semillas, extra=[])  # el resto de argumentos se pasa al script
    semillas.epilog = "Los argumentos que cfl no reconoce se pasan al script: cfl semillas domainnet.py --buffers 0 20"
    return parser


def main(argv=None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command == "semillas":
        args.extra = [arg for arg in unknown if arg != "--"]
    elif unknown and args.command != "tests":
        parser.error("argumentos no reconocidos: " + " ".join(unknown))
    elif args.command == "tests":
        args.extra = [arg for arg in [*(args.extra or []), *unknown] if arg != "--"]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
