"""Informe de un plan de varias tareas (cfl_tareas.py): resultados/<plan>/INFORME.md y gráficas en informe/.

    python informe_tareas.py planes/plan_tareas.json

Funciona con resultados parciales. Cifras: media ± desviación típica muestral entre semillas.
La precisión sobre las clases vistas tras cada tarea se obtiene de matriz.csv como la media de
las tareas vistas (todas tienen el mismo número de imágenes de test).
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from informe_plan import ACCENT, DARK, GREY, ROOT, fmt, number, plt, stats, table

METRICS = (("Final", "acc_final"), ("Incremental", "acc_incremental_media"), ("Olvido", "olvido_medio"),
           ("BWT", "bwt"), ("T1 final", "acc_t1_final"), ("Última final", "acc_last_final"),
           ("WA", "wa_acc_final"), ("cRT", "crt_acc_final"))
COLORS = (ACCENT, "#d9822b", "#3b9b6d", "#9b59b6", "#c0392b", DARK)


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def last_task_alias(row):
    """acc_t5_final (o acc_tN_final de la última tarea) también como acc_last_final."""
    keys = [k for k in row if k.startswith("acc_t") and k.endswith("_final") and k != "acc_t1_final"]
    return {**row, "acc_last_final": row[keys[0]] if keys else None}


def load(plan):
    out = ROOT / plan["salida"]
    data, matrix, joint = {}, {}, {}
    for job in plan["trabajos"]:
        for path in sorted((out / job["id"]).glob("seed_*/results.csv")):
            seed = int(path.parent.name.split("_")[1])
            for row in read_csv(path):
                data.setdefault(job["id"], {}).setdefault(int(row["buffer"]), {})[seed] = last_task_alias(row)
            if (path.parent / "matriz.csv").is_file():
                for row in read_csv(path.parent / "matriz.csv"):
                    cell = matrix.setdefault(job["id"], {}).setdefault(int(row["buffer"]), {}).setdefault(seed, {})
                    cell[(int(row["tras_tarea"]), int(row["tarea"]))] = float(row["acc"])
            if (path.parent / "joint_results.csv").is_file():
                joint.setdefault(job["id"], {})[seed] = read_csv(path.parent / "joint_results.csv")[0]
    return data, matrix, joint


def metric(data, job, buffer, key):
    return stats([number(r.get(key)) for r in data.get(job, {}).get(buffer, {}).values()])


def seen_curve(cells):
    """Precisión sobre las clases vistas tras cada tarea, para una semilla."""
    tasks = sorted({t for t, _ in cells})
    return [sum(cells[(t, i)] for i in range(1, t + 1)) / t for t in tasks]


def mean_matrix(seeds):
    keys = set.intersection(*(set(c) for c in seeds.values()))
    return {k: sum(c[k] for c in seeds.values()) / len(seeds) for k in keys}


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    out = ROOT / plan["salida"]
    figures = out / "informe"
    figures.mkdir(parents=True, exist_ok=True)
    titles = {j["id"]: j.get("titulo", j["id"]) for j in plan["trabajos"]}
    data, matrix, joint = load(plan)
    md = [f"# Informe del plan «{plan['nombre']}»", "",
          f"Generado el {datetime.now():%d/%m/%Y %H:%M}. {plan.get('descripcion', '')}", "",
          "Cifras: media ± desviación típica muestral entre semillas; n = número de semillas. Precisión en %.",
          "Final = todas las clases tras la última tarea; Incremental = media de la precisión sobre las clases vistas "
          "tras cada tarea; Olvido = caída media desde el mejor valor de cada tarea antigua; BWT = cambio medio de "
          "cada tarea antigua desde que se aprendió.", ""]

    status_path = out / "estado.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        counts = {s: sum(v == s for v in status.values()) for s in set(status.values())}
        md += ["## Estado", "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), ""]
        problems = [t for t, s in status.items() if s in ("fallida", "omitida")]
        if problems:
            md += ["Tareas con problemas (ver `_registros/<tarea>.log`): " + ", ".join(problems), ""]

    # ---------------------------------------------------------------- por trabajo
    md += ["## 1. Resultados por trabajo", ""]
    for job in plan["trabajos"]:
        jid = job["id"]
        if jid not in data:
            md += [f"**{titles[jid]}** (`{jid}`): sin resultados todavía.", ""]
            continue
        rows = [[f"{b} %", len(data[jid][b]), *[fmt(metric(data, jid, b, key)) for _, key in METRICS]]
                for b in sorted(data[jid])]
        md += [f"### {titles[jid]} (`{jid}`)", "", table(["Buffer", "n", *[name for name, _ in METRICS]], rows), ""]
        if jid in joint:
            j = joint[jid]
            md += [f"Referencia conjunta (n = {len(j)}): {fmt(stats([number(r['acc_final']) for r in j.values()]))}.", ""]

    # ---------------------------------------------------------------- comparación entre métodos
    buffers = sorted({b for jid in data for b in data[jid] if b > 0})
    if buffers:
        md += ["## 2. Comparación entre métodos", ""]
        headers = ["Método", *[f"{name} {b} %" for b in buffers for name in ("Final", "Olvido", "cRT")]]
        rows = [[titles[jid], *[fmt(metric(data, jid, b, key)) for b in buffers
                                for key in ("acc_final", "olvido_medio", "crt_acc_final")]]
                for jid in titles if jid in data]
        md += [table(headers, rows), ""]

    # ---------------------------------------------------------------- pareadas frente a la mezcla normal
    if "normal" in data:
        rows = []
        for jid in titles:
            if jid == "normal" or jid not in data:
                continue
            for b in buffers:
                pairs = [(number(r["acc_final"]), number(data["normal"].get(b, {})[s]["acc_final"]))
                         for s, r in data[jid].get(b, {}).items() if s in data["normal"].get(b, {})]
                if pairs:
                    diffs = [a - n for a, n in pairs]
                    rows.append([titles[jid], f"{b} %", len(diffs), fmt(stats(diffs)),
                                 f"{sum(d > 0 for d in diffs)} de {len(diffs)}"])
        if rows:
            md += ["## 3. Diferencias pareadas por semilla frente a la mezcla normal (precisión final)", "",
                   table(["Método", "Buffer", "n", "Diferencia (pp)", "Gana en"], rows), ""]

    # ---------------------------------------------------------------- gráficas
    reference_buffer = max(buffers) if buffers else None
    if reference_buffer is not None and matrix:
        fig, ax = plt.subplots(figsize=(7, 4))
        for color, jid in zip(COLORS, [j for j in titles if reference_buffer in matrix.get(j, {})]):
            curves = [seen_curve(c) for c in matrix[jid][reference_buffer].values()]
            mean = [sum(v) / len(v) for v in zip(*curves)]
            ax.plot(range(1, len(mean) + 1), mean, marker="o", color=color, label=titles[jid])
        baseline = next((matrix[j][0] for j in titles if 0 in matrix.get(j, {})), None)
        if baseline:
            curves = [seen_curve(c) for c in baseline.values()]
            ax.plot(range(1, len(curves[0]) + 1), [sum(v) / len(v) for v in zip(*curves)], ls="--", color=GREY,
                    label="Sin replay")
        for jid, seeds_ in joint.items():
            ax.axhline(sum(number(r["acc_final"]) for r in seeds_.values()) / len(seeds_), color=DARK, ls=":",
                       label="Referencia conjunta")
        ax.set(xlabel="Tras la tarea", ylabel="Precisión sobre las clases vistas (%)",
               title=f"Precisión incremental (buffer {reference_buffer} %)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "curva_incremental.png", dpi=150)
        plt.close(fig)
        md += ["## 4. Gráficas", "", "![Precisión incremental](informe/curva_incremental.png)", ""]

        jobs = [j for j in titles if reference_buffer in matrix.get(j, {})]
        fig, axes = plt.subplots(1, len(jobs), figsize=(3.2 * len(jobs), 3.2), squeeze=False)
        for ax, jid in zip(axes[0], jobs):
            cells = mean_matrix(matrix[jid][reference_buffer])
            n = max(t for t, _ in cells)
            grid = [[cells.get((t, i), float("nan")) if i <= t else float("nan") for i in range(1, n + 1)]
                    for t in range(1, n + 1)]
            ax.imshow(grid, cmap="Blues", vmin=0, vmax=100)
            for t in range(n):
                for i in range(t + 1):
                    ax.text(i, t, f"{grid[t][i]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[t][i] > 55 else DARK)
            ax.set(title=titles[jid][:28], xlabel="Tarea evaluada", ylabel="Tras la tarea",
                   xticks=range(n), xticklabels=range(1, n + 1), yticks=range(n), yticklabels=range(1, n + 1))
            ax.grid(False)
        fig.tight_layout()
        fig.savefig(figures / "matrices.png", dpi=150)
        plt.close(fig)
        md += [f"Matriz de precisión R[t][i] (media entre semillas, buffer {reference_buffer} %): precisión en la "
               "tarea i tras aprender la tarea t.", "", "![Matrices](informe/matrices.png)", ""]

    (out / "INFORME.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[INFORME] {out / 'INFORME.md'}")


if __name__ == "__main__":
    main(sys.argv[1])
