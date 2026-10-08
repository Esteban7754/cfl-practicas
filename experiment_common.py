"""Utilidades compartidas por los experimentos de replay.

Centraliza lo que antes estaba duplicado (y con errores) en cada script:
argumentos de línea de comandos, selección de buffers anidados, presupuesto
de entrenamiento, tablas, CSV y gráficos de pérdida de precisión.
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

try:  # matplotlib solo hace falta para los gráficos
    import matplotlib
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_BUFFERS = [0, 5, 10, 15, 20, 25]
COLORS = ["#e74c3c", "#3498db", "#f1c40f", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#34495e"]


def build_parser(description, default_buffers=None):
    """Argumentos comunes a todos los experimentos de replay."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, help="Semilla (por defecto la del CONFIG). Repite con varias para medir la variabilidad")
    parser.add_argument("--buffers", nargs="+", type=int, default=list(default_buffers or DEFAULT_BUFFERS),
                        help="Porcentajes de replay a comparar; 0 = sin replay (siempre se mide, nunca se fija a mano)")
    parser.add_argument("--rounds", type=int, help="Sobrescribe el número de rondas por fase")
    parser.add_argument("--local-epochs", type=int, help="Sobrescribe las épocas locales")
    parser.add_argument("--equal-steps", action=argparse.BooleanOptionalAction, default=True,
                        help="Mismo número de actualizaciones en todas las variantes (por defecto): el replay sustituye parte "
                             "de los lotes de B en lugar de añadir pasos. --no-equal-steps recupera el diseño antiguo, en el "
                             "que más buffer también implica más entrenamiento")
    parser.add_argument("--output-dir", help="Carpeta para resultados (CSV, config y gráficos)")
    return parser


def apply_args(args, config, parser):
    """Valida los argumentos, actualiza CONFIG y prepara la carpeta de salida."""
    if any(p < 0 or p > 100 for p in args.buffers) or len(args.buffers) != len(set(args.buffers)):
        parser.error("--buffers debe contener porcentajes entre 0 y 100 sin repetir")
    if 0 not in args.buffers:
        parser.error("--buffers debe incluir 0: la referencia sin replay tiene que medirse")
    args.buffers = sorted(args.buffers)
    if args.seed is not None:
        config["seed"] = args.seed
    for value, key in ((args.rounds, "num_rounds"), (args.local_epochs, "local_epochs")):
        if value is not None:
            if value < 1:
                parser.error("Las rondas y épocas deben ser positivas")
            config[key] = value
    config["equal_steps"] = bool(args.equal_steps)
    config["buffers"] = args.buffers
    if not args.output_dir:
        # Nunca escribir en la raíz: así no se sobrescriben los gráficos históricos.
        script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.output_dir = os.path.join("resultados", f"{script}_seed{config['seed']}_{stamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    print(f"[INFO] Resultados en: {os.getcwd()}")
    try:
        from reproducibility import environment_info
        environment = environment_info()
    except ImportError:  # pragma: no cover - sin PyTorch
        environment = None
    with open("config.json", "w", encoding="utf-8") as handle:
        json.dump({**config, "environment": environment}, handle, indent=2, default=str)
    return config


def buffer_label(percent):
    return "No Replay" if percent == 0 else f"{percent}%"


def nested_buffer_indices(n_samples, fraction, seed, client_id):
    """Índices del buffer de un cliente.

    Usa un generador propio (no consume el estado global) y la misma
    permutación para todos los porcentajes, de modo que el buffer del 5 % está
    contenido en el del 10 %, etc. Así las variantes solo difieren en la
    capacidad de memoria y no dependen del orden de ejecución.
    """
    size = int(n_samples * fraction)
    if size <= 0:
        return []
    order = np.random.default_rng(seed * 1000 + client_id).permutation(n_samples)
    return [int(i) for i in order[:size]]


def step_budget(n_new_samples, batch_size, epochs, equal_steps):
    """Máximo de actualizaciones por cliente y ronda en la fase 2.

    Con ``equal_steps`` todas las variantes hacen las mismas actualizaciones que
    la variante sin replay. Sin él (diseño original) el replay añade pasos.
    """
    if not equal_steps:
        return None
    return math.ceil(n_new_samples / batch_size) * epochs


def make_result(percent, acc_a_before, acc_a_after, acc_b_after, **extra):
    return {
        "replay_percent": percent,
        "fraction": buffer_label(percent),
        "acc_a_before": acc_a_before,
        "acc_a_after": acc_a_after,
        "acc_b_after": acc_b_after,
        "loss_a_pp": acc_a_before - acc_a_after,
        **extra,
    }


def print_table(title, results, group_a="Group A", group_b="Group B"):
    width = 100
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"{'Buffer':<12} | {group_a + ' antes':>16} | {group_a + ' después':>18} | {group_b + ' después':>18} | {'Pérdida A':>12}")
    print("-" * width)
    for r in results:
        print(f"{r['fraction']:<12} | {r['acc_a_before']:>15.2f}% | {r['acc_a_after']:>17.2f}% | "
              f"{r['acc_b_after']:>17.2f}% | {r['loss_a_pp']:>9.2f} pp")
    print("=" * width)
    print("Pérdida A = precisión A antes - precisión A después, en puntos porcentuales (pp).")


def save_results_csv(results, path="results.csv"):
    if not results:
        return
    fields = list(dict.fromkeys(key for r in results for key in r))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Resultados guardados en '{path}'")


def plot_loss(results, filename, title, xlabel="Replay buffer", ylabel="Pérdida de precisión en A (pp)", errors=None):
    """Gráfico de barras que admite valores negativos y barras de error."""
    if plt is None or not results:
        return
    labels = [r["fraction"] for r in results]
    values = [r["loss_a_pp"] for r in results]
    colors = [COLORS[i % len(COLORS)] for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=1,
                  yerr=errors, capsize=6 if errors is not None else 0)
    ax.axhline(0, color="black", linewidth=0.8)
    span = max(max(values), 0) - min(min(values), 0)
    pad = max(span * 0.08, 2)
    top = max(max(values), 0) + (max(errors) if errors else 0)
    bottom = min(min(values), 0) - (max(errors) if errors else 0)
    ax.set_ylim(bottom - (pad if bottom < 0 else 0), top + pad * 1.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, (bar, value) in enumerate(zip(bars, values)):
        err = errors[i] if errors else 0
        offset = pad * 0.25 + err
        y = value + offset if value >= 0 else value - offset
        text = f"{value:.2f} pp" if errors is None else f"{value:.2f} ± {errors[i]:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, y, text, ha="center",
                va="bottom" if value >= 0 else "top", fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"[INFO] Gráfico guardado en '{filename}'")
