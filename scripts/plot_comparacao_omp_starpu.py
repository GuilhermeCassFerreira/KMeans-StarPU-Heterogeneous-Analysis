"""Comparação StarPU+MPI híbrido × OpenMP+MPI híbrido nos cenários PQ, Deep e SIFT, de 1 a 32 nós.

Lê o "Tempo do loop" (rank mais lento) das 5 medições de cada configuração:
  StarPU: results/experiments/escala/<cenario>_escala_np<N>/results.csv
  OpenMP: results/experiments/omp_escala/omp_hibrido_np<N>/results.csv
          (8 nós: results/experiments/omp_escala_np8/omp_hibrido_np8/results.csv)
e gera, em results/figures/:
  comparacao_omp_starpu.csv, comparacao_tabela_tempos.png, comparacao_tabela_speedup.png,
  comparacao_omp_starpu.png/.pdf
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "experiments")
OUT = os.path.join(ROOT, "results", "figures")
NODES = [1, 2, 4, 8, 16, 32]
SCENARIOS = [("pq", "PQ", "N = 1,024×10⁸, D = 16"), ("deep", "Deep", "N = 2,048×10⁷, D = 96"),
             ("sift", "SIFT", "N = 1,024×10⁷, D = 128")]
STARPU, OMP = "#2a78d6", "#eb6834"
GRAY, INK, MUTED, GRID = "#52514e", "#0b0b0b", "#8a8984", "#e6e5e1"


def fmt(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def starpu_times(key, n):
    path = os.path.join(EXP, "escala", f"{key}_escala_np{n}", "results.csv")
    return [float(r["loop_ms"]) / 1000 for r in csv.DictReader(open(path))
            if r["rep_kind"] == "metricas" and r["run_ok"] == "1"]


def omp_times(key, n):
    path = (os.path.join(EXP, "omp_escala_np8", "omp_hibrido_np8", "results.csv") if n == 8 else
            os.path.join(EXP, "omp_escala", f"omp_hibrido_np{n}", "results.csv"))
    return [float(r["loop_ms"]) / 1000 for r in csv.DictReader(open(path))
            if r["cenario"] == key and r["run_ok"] == "1"]


def load():
    data = {}
    for key, *_ in SCENARIOS:
        for impl, fn in (("starpu", starpu_times), ("omp", omp_times)):
            for n in NODES:
                t = fn(key, n)
                if len(t) != 5:
                    raise SystemExit(f"{impl} {key} {n} nós: esperava 5 medições, achei {len(t)}")
                data[impl, key, n] = dict(mean=statistics.mean(t), sd=statistics.stdev(t))
            for n in NODES:
                d = data[impl, key, n]
                d["sp"] = data[impl, key, 1]["mean"] / d["mean"]
    return data


def diff_pct(data, key, n):
    """> 0: StarPU mais rápido por esse percentual; < 0: OpenMP mais rápido."""
    s, o = data["starpu", key, n]["mean"], data["omp", key, n]["mean"]
    return (o / s - 1) * 100 if o >= s else -(s / o - 1) * 100


def write_csv(data):
    with open(os.path.join(OUT, "comparacao_omp_starpu.csv"), "w") as f:
        f.write("cenario,nos,starpu_s,starpu_desvio_s,omp_s,omp_desvio_s,starpu_speedup,omp_speedup,vencedor,diferenca_pct\n")
        for key, *_ in SCENARIOS:
            for n in NODES:
                s, o = data["starpu", key, n], data["omp", key, n]
                d = diff_pct(data, key, n)
                f.write(f"{key},{n},{s['mean']:.4f},{s['sd']:.4f},{o['mean']:.4f},{o['sd']:.4f},"
                        f"{s['sp']:.3f},{o['sp']:.3f},{'starpu' if d > 0 else 'omp'},{abs(d):.2f}\n")


def table_png(data, out, title, value, note, menor_melhor):
    """value(d) -> (texto, número comparado); em negrito, o melhor dos dois (empate: nenhum)."""
    header = ["Nós"]
    for _, name, _ in SCENARIOS:
        header += [f"{name}\nStarPU", f"{name}\nOpenMP"]
    rows, bold = [], set()
    for i, n in enumerate(NODES, start=1):
        row = [str(n)]
        for j, (key, *_) in enumerate(SCENARIOS):
            (ts, vs), (to, vo) = value(data["starpu", key, n]), value(data["omp", key, n])
            row += [ts, to]
            if vs != vo:
                starpu_melhor = (vs < vo) if menor_melhor else (vs > vo)
                bold.add((i, 1 + 2 * j + (0 if starpu_melhor else 1)))
        rows.append(row)
    fig = plt.figure(figsize=(11.6, 4.3))
    ax = fig.add_axes([0.01, 0.17, 0.98, 0.72])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1],
                  colWidths=[0.07] + [0.155] * 6)
    tb.auto_set_font_size(False)
    tb.set_fontsize(11.5)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#9a9995")
        if i == 0:
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK, weight="bold" if (i, j) in bold else "normal")
    fig.text(0.01, 0.93, title, fontsize=13, color=INK)
    fig.text(0.01, 0.02, note, fontsize=9, color=GRAY)
    fig.savefig(os.path.join(OUT, out), dpi=200, bbox_inches="tight")
    plt.close(fig)


def chart(data):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": GRAY, "ytick.color": GRAY})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), gridspec_kw={"height_ratios": [1.15, 1]})
    for col, (key, name, desc) in enumerate(SCENARIOS):
        # Linha 1: speedup sobre 1 nó de cada versão, com o ideal
        ax = axes[0][col]
        ax.plot(NODES, NODES, "--", color=GRAY, linewidth=1.3, label="Ideal (linear)")
        finais = {impl: data[impl, key, 32]["sp"] for impl in ("starpu", "omp")}
        for impl, label, color, marker in (("starpu", "StarPU", STARPU, "o"), ("omp", "OpenMP", OMP, "s")):
            sp = [data[impl, key, n]["sp"] for n in NODES]
            ax.plot(NODES, sp, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor="white", markeredgewidth=1, label=label)
            dy = 8 if finais[impl] >= max(finais.values()) else -8  # o maior em cima, o menor embaixo
            ax.annotate(f"{label} {fmt(sp[-1], 1)}×", (32, sp[-1]), xytext=(7, dy), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(NODES, [str(n) for n in NODES])
        ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.set_xlim(0.85, 120)
        ax.set_title(f"{name} ({desc})", loc="left", fontsize=10, color=INK)
        if col == 0:
            ax.set_ylabel("Speedup sobre 1 nó")
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")

        # Linha 2: diferença de tempo entre as duas versões em cada número de nós
        ax = axes[1][col]
        diffs = [diff_pct(data, key, n) for n in NODES]
        xs = range(len(NODES))
        ax.bar(xs, diffs, width=0.62, color=[STARPU if d > 0 else OMP for d in diffs])
        for x, d in zip(xs, diffs):
            ax.text(x, d + (1 if d >= 0 else -1), f"{fmt(abs(d), 0)}%", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8.5, color=INK)
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.set_xticks(list(xs), [str(n) for n in NODES])
        ax.set_xlabel("Número de nós")
        lim = max(30, max(abs(d) for d in diffs) + 6)
        ax.set_ylim(-lim, lim)
        if col == 0:
            ax.set_ylabel("Quanto a versão mais rápida\nganha (%)")
        ax.text(0.01, 0.97, "StarPU mais rápido ↑", transform=ax.transAxes, fontsize=8.5, color=STARPU, va="top")
        ax.text(0.01, 0.03, "OpenMP mais rápido ↓", transform=ax.transAxes, fontsize=8.5, color=OMP, va="bottom")

    for row in axes:
        for ax in row:
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle("StarPU+MPI híbrido × OpenMP+MPI híbrido — escalabilidade de 1 a 32 nós", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, -0.01, "Média de 5 execuções por configuração; tempo de loop do rank mais lento; K = 256, 30 iterações, "
             "AWS g6.4xlarge (6 núcleos de CPU + 1 GPU L4 por nó). Speedup de cada versão sobre o próprio tempo com 1 nó. "
             "StarPU e OpenMP rodaram em clusters diferentes (3–7% de variação entre clusters).", fontsize=7.5,
             color=GRAY, wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"comparacao_omp_starpu.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load()
    write_csv(data)
    note = ("Média de 5 execuções; tempo de loop do rank mais lento; K = 256, 30 iterações, AWS g6.4xlarge "
            "(6 núcleos de CPU + 1 GPU L4 por nó). Em negrito, a versão mais rápida.\nStarPU: dmda, 1024 chunks. "
            "OpenMP: gpu_ratio 0,85 (fórmula), reduce e broadcast lineares. As duas rodaram em clusters diferentes.")
    table_png(data, "comparacao_tabela_tempos.png", "Tempo de loop (s) — StarPU+MPI híbrido × OpenMP+MPI híbrido",
              lambda d: (f"{fmt(d['mean'], 2)} ± {fmt(d['sd'], 2)}", round(d["mean"], 3)), note, menor_melhor=True)
    table_png(data, "comparacao_tabela_speedup.png", "Speedup sobre 1 nó — StarPU+MPI híbrido × OpenMP+MPI híbrido",
              lambda d: (f"{fmt(d['sp'], 1)}×", round(d["sp"], 1)),
              note.replace("Em negrito, a versão mais rápida.", "Em negrito, o maior speedup.")
                  .replace("(6 núcleos", "(cada versão sobre o próprio tempo com 1 nó; 6 núcleos"),
              menor_melhor=False)
    chart(data)
    for key, name, _ in SCENARIOS:
        print(name.ljust(5), "  ".join(f"{n}:{diff_pct(data, key, n):+.0f}%" for n in NODES), "(+ = StarPU mais rápido)")
