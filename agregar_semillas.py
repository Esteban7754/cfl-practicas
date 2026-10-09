"""Combina los results.csv de varias semillas: media, desviación típica y gráfico.

Uso:
    python agregar_semillas.py resultados/mi_experimento/seed_42 resultados/mi_experimento/seed_43 ...
    python agregar_semillas.py resultados/mi_experimento          # busca seed_*/results.csv dentro

Acepta tanto el formato de experiment_common (loss_a_pp, acc_a_after...) como el de
cifar100_three_buffer.py (loss_a, step4_acc_a...).
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from experiment_common import plot_loss, save_results_csv

ALIASES = {
    "acc_a_before": ("acc_a_before", "step3_acc_a", "p1_acc_a"),
    "acc_a_after": ("acc_a_after", "step4_acc_a", "p2_acc_a"),
    "acc_b_after": ("acc_b_after", "step4_acc_b", "p2_acc_b"),
    "loss_a_pp": ("loss_a_pp", "loss_a"),
}
# Columnas que solo tienen las ejecuciones recientes: se agregan si están en todos los CSV.
OPTIONAL = ("retention_a_percent", "acc_all_100_classes", "avg_acc_tasks", "bwt_a",
            "wa_acc_a", "wa_acc_b", "wa_acc_all_100_classes", "wa_retention_a_percent",
            "gap_all_vs_joint", "intransigence_b")
JOINT = ("acc_a", "acc_b", "acc_all_100_classes", "avg_acc_tasks")


def find_csvs(paths):
    found = []
    for path in map(Path, paths):
        if path.is_file():
            found.append(path)
        elif (path / "results.csv").is_file():
            found.append(path / "results.csv")
        else:
            found.extend(sorted(path.glob("seed_*/results.csv")))
    return found


def replay_percent(row):
    if row.get("replay_percent") not in (None, ""):
        return int(float(row["replay_percent"]))
    label = row.get("fraction", "")
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 0


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values = {}
            for key, options in ALIASES.items():
                source = next((o for o in options if row.get(o) not in (None, "")), None)
                if source is None:
                    raise ValueError(f"{csv_path}: falta la columna {key}")
                values[key] = float(row[source])
            for key in OPTIONAL:
                if row.get(key) not in (None, ""):
                    values[key] = float(row[key])
            yield replay_percent(row), values


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="results.csv, carpetas de semilla o la carpeta que contiene seed_*")
    parser.add_argument("--output-dir", type=Path, help="Dónde guardar el resumen (por defecto, la primera carpeta indicada)")
    parser.add_argument("--title", default="Pérdida de precisión en A (media ± desviación típica)")
    args = parser.parse_args()

    csv_paths = find_csvs(args.paths)
    if not csv_paths:
        parser.error("No se encontró ningún results.csv")

    grouped = defaultdict(lambda: defaultdict(list))
    for csv_path in csv_paths:
        for percent, values in read_rows(csv_path):
            for key, value in values.items():
                grouped[percent][key].append(value)

    counts = {percent: len(metrics["loss_a_pp"]) for percent, metrics in grouped.items()}
    if len(set(counts.values())) > 1:
        print(f"[AVISO] No todos los buffers tienen el mismo número de semillas: {counts}")

    summary = []
    for percent in sorted(grouped):
        row = {"replay_percent": percent, "fraction": "No Replay" if percent == 0 else f"{percent}%",
               "n_seeds": counts[percent]}
        for key, values in grouped[percent].items():
            if len(values) == counts[percent]:  # una métrica opcional que falta en alguna semilla no se promedia
                row[f"{key}_mean"], row[f"{key}_std"] = mean_std(values)
        summary.append(row)

    output_dir = args.output_dir or (Path(args.paths[0]) if Path(args.paths[0]).is_dir() else Path(args.paths[0]).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv(summary, output_dir / "resumen_semillas.csv")

    print(f"\n{'Buffer':<10} | {'n':>2} | {'A después':>16} | {'B después':>16} | {'Pérdida A (pp)':>18}")
    print("-" * 74)
    for r in summary:
        print(f"{r['fraction']:<10} | {r['n_seeds']:>2} | {r['acc_a_after_mean']:>7.2f} ± {r['acc_a_after_std']:<6.2f} | "
              f"{r['acc_b_after_mean']:>7.2f} ± {r['acc_b_after_std']:<6.2f} | {r['loss_a_pp_mean']:>8.2f} ± {r['loss_a_pp_std']:<6.2f}")
    if all("wa_acc_a_mean" in r for r in summary):
        print(f"\n{'Buffer':<10} | {'Retención A':>16} | {'100 clases':>16} | {'A con WA':>16} | {'100 clases WA':>16}")
        print("-" * 86)
        for r in summary:
            retention = (f"{r['retention_a_percent_mean']:>7.2f} ± {r['retention_a_percent_std']:<6.2f}"
                         if "retention_a_percent_mean" in r else f"{'—':>16}")
            print(f"{r['fraction']:<10} | {retention} | {r['acc_all_100_classes_mean']:>7.2f} ± {r['acc_all_100_classes_std']:<6.2f} | "
                  f"{r['wa_acc_a_mean']:>7.2f} ± {r['wa_acc_a_std']:<6.2f} | "
                  f"{r['wa_acc_all_100_classes_mean']:>7.2f} ± {r['wa_acc_all_100_classes_std']:<6.2f}")

    joint_rows = []
    for csv_path in csv_paths:
        joint_path = csv_path.parent / "joint_results.csv"
        if joint_path.is_file():
            with open(joint_path, newline="", encoding="utf-8") as handle:
                joint_rows.extend(csv.DictReader(handle))
    if joint_rows:
        joint = {"n_seeds": len(joint_rows)}
        for key in JOINT:
            joint[f"{key}_mean"], joint[f"{key}_std"] = mean_std([float(r[key]) for r in joint_rows])
        save_results_csv([joint], output_dir / "resumen_conjunto.csv")
        print(f"\nCota superior (conjunto A+B, {len(joint_rows)} semillas): "
              f"A {joint['acc_a_mean']:.2f} ± {joint['acc_a_std']:.2f} | B {joint['acc_b_mean']:.2f} ± {joint['acc_b_std']:.2f} | "
              f"100 clases {joint['acc_all_100_classes_mean']:.2f} ± {joint['acc_all_100_classes_std']:.2f}")

    if min(counts.values()) < 3:
        print("\n[AVISO] Con menos de 3 semillas la desviación típica es poco fiable.")

    plot_rows = [{"fraction": r["fraction"], "loss_a_pp": r["loss_a_pp_mean"]} for r in summary]
    plot_loss(plot_rows, str(output_dir / "perdida_A_media_semillas.png"), args.title,
              errors=[r["loss_a_pp_std"] for r in summary])


if __name__ == "__main__":
    main()
