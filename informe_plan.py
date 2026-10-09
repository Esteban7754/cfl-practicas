"""Informe automático de un plan de experimentos: resultados/<plan>/INFORME.md y gráficas en informe/.

    python informe_plan.py planes/plan_gpu.json

Funciona con resultados parciales: cada tabla o gráfica usa lo que haya y lo dice.
Cifras: media ± desviación típica muestral entre semillas (n indicado en cada tabla).
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
ACCENT, GREY, DARK = "#2a78d6", "#a7a59e", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False})


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def buffer_of(row):
    return 0 if row["fraction"] == "No Replay" else int(row["fraction"].split("%")[0])


def load(plan):
    out = ROOT / plan["salida"]
    data, joint = {}, {}
    for job in plan["trabajos"]:
        for path in sorted((out / job["id"]).glob("seed_*/results.csv")):
            seed = int(path.parent.name.split("_")[1])
            with open(path, encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    data.setdefault(job["id"], {}).setdefault(buffer_of(row), {})[seed] = row
            joint_path = path.parent / "joint_results.csv"
            if joint_path.is_file():
                with open(joint_path, encoding="utf-8") as handle:
                    joint.setdefault(job["id"], {})[seed] = next(csv.DictReader(handle))
    return data, joint


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd, len(values)


def metric(data, job, buffer, key):
    rows = data.get(job, {}).get(buffer, {})
    return stats([number(r.get(key)) for r in rows.values()])


def fmt(s, decimals=1):
    if s is None:
        return "—"
    mean, sd, _ = s
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}".replace(".", ",")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def seeds(data, job, buffer=0):
    return len(data.get(job, {}).get(buffer, {}))


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    out = ROOT / plan["salida"]
    figures = out / "informe"
    figures.mkdir(parents=True, exist_ok=True)
    titles = {j["id"]: j.get("titulo", j["id"]) for j in plan["trabajos"]}
    data, joint = load(plan)
    md = [f"# Informe del plan «{plan['nombre']}»", "",
          f"Generado el {datetime.now():%d/%m/%Y %H:%M}. {plan.get('descripcion', '')}", "",
          "Cifras: media ± desviación típica muestral entre semillas; n = número de semillas. Precisión en %.", ""]

    status_path = out / "estado.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        counts = {s: sum(v == s for v in status.values()) for s in set(status.values())}
        md += ["## Estado", "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), ""]
        problems = [t for t, s in status.items() if s in ("fallida", "omitida")]
        if problems:
            md += ["Tareas con problemas (ver `_registros/<tarea>.log`): " + ", ".join(problems), ""]

    # ---------------------------------------------------------------- tabla general por trabajo
    md += ["## 1. Resultados por trabajo", ""]
    for job in plan["trabajos"]:
        jid = job["id"]
        if jid not in data:
            md += [f"**{titles[jid]}** (`{jid}`): sin resultados todavía.", ""]
            continue
        rows = []
        for b in sorted(data[jid]):
            rows.append([f"{b} %", seeds(data, jid, b), fmt(metric(data, jid, b, "step4_acc_a")),
                         fmt(metric(data, jid, b, "step4_acc_b")), fmt(metric(data, jid, b, "retention_a_percent")),
                         fmt(metric(data, jid, b, "acc_all_100_classes")), fmt(metric(data, jid, b, "wa_acc_all_100_classes")),
                         fmt(metric(data, jid, b, "crt_acc_all_100_classes"))])
        md += [f"**{titles[jid]}** (`{jid}`)", "",
               table(["Buffer", "n", "A final", "B final", "Retención A", "100 clases", "100 cl. con WA", "100 cl. con cRT"], rows), ""]
        if jid in joint:
            j = joint[jid]
            md += [f"Referencia conjunta ({len(j)} semillas, presupuesto x{number(next(iter(j.values())).get('budget_factor')) or 1:g}): "
                   f"A {fmt(stats([number(r['acc_a']) for r in j.values()]))}, B {fmt(stats([number(r['acc_b']) for r in j.values()]))}, "
                   f"100 clases {fmt(stats([number(r['acc_all_100_classes']) for r in j.values()]))}.", ""]

    # ---------------------------------------------------------------- 2. fracción por lote
    fractions = [("Mezcla normal", "principal_normal"), ("natural", "fraccion_natural"), ("0,25", "fraccion_025"),
                 ("0,5", "principal_equilibrados"), ("0,75", "fraccion_075")]
    available = [(label, jid) for label, jid in fractions if jid in data]
    if len(available) >= 3:
        md += ["## 2. Ablación de la fracción del lote", "",
               "¿Pesa más la proporción del buffer en cada lote o el muestreo con reposición? «natural» usa lotes "
               "equilibrados con la misma proporción que tendría la mezcla normal: si retiene como la mezcla normal, "
               "lo decisivo es la proporción; si retiene como 0,5, es el muestreo.", ""]
        rows = [[label, *[fmt(metric(data, jid, b, key)) for b in (10, 20) for key in ("retention_a_percent", "acc_all_100_classes")]]
                for label, jid in available]
        md += [table(["Variante", "Retención (10 %)", "100 clases (10 %)", "Retención (20 %)", "100 clases (20 %)"], rows), ""]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, key, name in ((axes[0], "retention_a_percent", "Retención de A (%)"), (axes[1], "acc_all_100_classes", "100 clases (%)")):
            for b, color in ((10, GREY), (20, ACCENT)):
                pts = [(label, metric(data, jid, b, key)) for label, jid in available]
                pts = [(l, s) for l, s in pts if s]
                ax.errorbar([l for l, _ in pts], [s[0] for _, s in pts], yerr=[s[1] for _, s in pts], color=color,
                            marker="o", capsize=3, lw=2, label=f"buffer {b} %")
            ax.set_title(name)
            ax.legend()
        fig.suptitle("Ablación de la fracción del lote que sale del buffer", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figures / "fraccion.png", dpi=150)
        plt.close(fig)
        md += ["![Ablación de la fracción](informe/fraccion.png)", ""]

    # ---------------------------------------------------------------- 3. correcciones del clasificador
    variants = [(titles.get(j, j), j) for j in ("principal_normal", "principal_equilibrados", "principal_lwf", "er_ace") if j in data]
    if variants:
        md += ["## 3. Correcciones del clasificador: WA, sus variantes y cRT", "",
               "Precisión en las 100 clases del mismo modelo final, sin corrección y con cada corrección posterior.", ""]
        kinds = [("Sin corrección", "acc_all_100_classes"), ("WA", "wa_acc_all_100_classes"),
                 ("WA solo pesos", "wa_w_acc_all_100_classes"), ("WA solo sesgo", "wa_b_acc_all_100_classes"),
                 ("cRT", "crt_acc_all_100_classes")]
        rows = [[f"{label} · {b} %", *[fmt(metric(data, jid, b, key)) for _, key in kinds]]
                for label, jid in variants for b in (5, 10, 20) if b in data[jid]]
        md += [table(["Variante · buffer", *[k for k, _ in kinds]], rows), ""]
        fig, ax = plt.subplots(figsize=(11, 4))
        short = {"principal_normal": "Normal", "principal_equilibrados": "Equilibrados", "principal_lwf": "Eq. + LwF",
                 "er_ace": "Eq. + ER-ACE"}
        groups = [(f"{short.get(jid, label)}\n{b} %", jid, b) for label, jid in variants for b in (10, 20) if b in data[jid]]
        width = 0.16
        shades = [GREY, ACCENT, "#86b6ef", "#184f95", "#eb6834"]
        for k, ((name, key), color) in enumerate(zip(kinds, shades)):
            vals = [metric(data, jid, b, key) for _, jid, b in groups]
            ax.bar([i + (k - 2) * width for i in range(len(groups))], [v[0] if v else 0 for v in vals], width,
                   yerr=[v[1] if v else 0 for v in vals], color=color, label=name, capsize=2)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        ax.set_ylabel("100 clases (%)")
        ax.set_title("Correcciones posteriores del clasificador")
        ax.legend(ncol=5, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(figures / "correcciones.png", dpi=150)
        plt.close(fig)
        md += ["![Correcciones del clasificador](informe/correcciones.png)", ""]

        md += ["### Diagnóstico de logits sobre el test de A", "",
               "Logit media de las clases A y de las B sobre imágenes de A, antes y después de WA. "
               "Sirve para contrastar la hipótesis de que WA funciona empujando hacia abajo logits B negativas.", ""]
        keys = [("A", "mean_logit_a_test_a"), ("B", "mean_logit_b_test_a"), ("A con WA", "wa_mean_logit_a_test_a"),
                ("B con WA", "wa_mean_logit_b_test_a"), ("γ", "wa_gamma")]
        rows = [[f"{label} · {b} %", *[fmt(metric(data, jid, b, key), 2) for _, key in keys]]
                for label, jid in variants for b in sorted(data[jid])]
        md += [table(["Variante · buffer", *[k for k, _ in keys]], rows), ""]

    # ---------------------------------------------------------------- 4. calendario adaptativo
    if "calendario" in data:
        md += ["## 4. Replay adaptativo en el tiempo", "",
               "Calendario 0,75-0,5-0,25-0,25-0,25 (media 0,40) frente a fracciones constantes. Si retiene como 0,5 "
               "con menos replay medio, el replay concentrado al principio es más eficiente.", ""]
        cands = [("Constante 0,25", "fraccion_025"), ("Calendario (media 0,40)", "calendario"), ("Constante 0,5", "principal_equilibrados")]
        rows = [[label, *[fmt(metric(data, jid, b, key)) for b in (10, 20) for key in ("retention_a_percent", "step4_acc_b", "acc_all_100_classes")]]
                for label, jid in cands if jid in data]
        md += [table(["Variante", "Retención (10 %)", "B (10 %)", "100 cl. (10 %)", "Retención (20 %)", "B (20 %)", "100 cl. (20 %)"], rows), ""]

    # ---------------------------------------------------------------- 5. referencia conjunta
    if joint:
        md += ["## 5. Referencia conjunta según el presupuesto", ""]
        rows = []
        for jid, per_seed in joint.items():
            factor = number(next(iter(per_seed.values())).get("budget_factor")) or 1
            rows.append([f"x{factor:g}", len(per_seed), *[fmt(stats([number(r[k]) for r in per_seed.values()]))
                                                        for k in ("acc_a", "acc_b", "acc_all_100_classes")]])
        md += [table(["Presupuesto", "n", "A", "B", "100 clases"], sorted(rows)), ""]

    # ---------------------------------------------------------------- 6. no IID
    settings = [("IID", "principal_normal", "principal_equilibrados"), ("Dirichlet 0,5", "dirichlet05_normal", "dirichlet05_equilibrados"),
                ("Dirichlet 0,1", "dirichlet01_normal", "dirichlet01_equilibrados")]
    settings = [s for s in settings if s[1] in data or s[2] in data]
    if len(settings) >= 2:
        md += ["## 6. Datos desiguales entre clientes (Dirichlet)", "",
               "Menor α = cada cliente tiene menos clases. ¿Siguen ganando los lotes equilibrados cuando los buffers locales están desequilibrados?", ""]
        rows = []
        for name, normal, balanced in settings:
            for label, jid in (("normal", normal), ("equilibrados", balanced)):
                if jid in data:
                    rows.append([f"{name} · {label}", fmt(metric(data, jid, 0, "acc_all_100_classes")),
                                 *[fmt(metric(data, jid, b, k)) for b in (10, 20) for k in ("retention_a_percent", "acc_all_100_classes")]])
        md += [table(["Reparto · variante", "100 cl. (0 %)", "Retención (10 %)", "100 cl. (10 %)", "Retención (20 %)", "100 cl. (20 %)"], rows), ""]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for k, (label, idx, color) in enumerate((("Mezcla normal", 1, GREY), ("Lotes equilibrados", 2, ACCENT))):
            vals = [metric(data, s[idx], 20, "retention_a_percent") for s in settings]
            ax.bar([i + (k - 0.5) * 0.35 for i in range(len(settings))], [v[0] if v else 0 for v in vals], 0.35,
                   yerr=[v[1] if v else 0 for v in vals], color=color, label=label, capsize=3)
        ax.set_xticks(range(len(settings)), [s[0] for s in settings])
        ax.set_ylabel("Retención de A con buffer 20 % (%)")
        ax.set_title("Retención según el reparto de datos entre clientes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "dirichlet.png", dpi=150)
        plt.close(fig)
        md += ["![Reparto Dirichlet](informe/dirichlet.png)", ""]

    (out / "INFORME.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[INFORME] {out / 'INFORME.md'} ({len(list(figures.glob('*.png')))} gráficas)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "planes/plan_gpu.json")
