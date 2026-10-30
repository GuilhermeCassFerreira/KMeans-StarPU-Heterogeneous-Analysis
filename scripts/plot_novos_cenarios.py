"""Tabela e gráfico de escalabilidade do StarPU+MPI híbrido nos cenários PQ, Deep e SIFT (1 e 32 nós).

Lê o "Tempo do loop" (rank mais lento) das 3 medições de cada configuração em
results/experiments/<cenario>_np{1,32}/results.csv e gera, em results/figures/:
  novos_cenarios.csv, novos_cenarios_tabela.png, novos_cenarios_escala.png/.pdf
O cenário do artigo (N=3e8, K=50, D=2) entra como referência (scripts/plot_escala_starpu.py).
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

SCENARIOS = [  # chave, nome, N, N formatado, K, D, cor, marcador, deslocamento do rótulo final (pt)
    ("pq", "Subvetores de PQ", 102400000, "1,024×10⁸", 256, 16, "#eb6834", "^", -11),
    ("deep", "Descritores Deep", 20480000, "2,048×10⁷", 256, 96, "#1baf7a", "s", 0),
    ("sift", "Descritores SIFT", 10240000, "1,024×10⁷", 256, 128, "#eda100", "D", 11),
]
# Cenário do artigo, medido em 1, 2, 4 e 32 nós (results/figures/starpu_hibrido_escala.csv)
ATUAL = {1: (15.4307, 0.001), 2: (7.7648, 0.006), 4: (3.9654, 0.027), 32: (1.10165, 0.133)}
BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#8a8984", "#e6e5e1"


def fmt(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def loop_times(key, nodes):
    path = os.path.join(EXP, f"{key}_np{nodes}", "results.csv")
    t = [float(r["loop_ms"]) / 1000 for r in csv.DictReader(open(path)) if r["rep_kind"] == "metricas"]
    if len(t) != 3:
        raise SystemExit(f"{path}: esperava 3 medições, achei {len(t)}")
    return statistics.mean(t), statistics.stdev(t)


def load():
    data = []
    for key, name, n_pts, n_txt, k, d, color, marker, dy in SCENARIOS:
        t1, sd1 = loop_times(key, 1)
        t32, sd32 = loop_times(key, 32)
        data.append(dict(key=key, name=name, n_pts=n_pts, n=n_txt, k=k, d=d, color=color, marker=marker, dy=dy,
                         t1=t1, sd1=sd1, t32=t32, sd32=sd32, sp=t1 / t32))
    return data


def write_csv(data):
    with open(os.path.join(OUT, "novos_cenarios.csv"), "w") as f:
        f.write("cenario,N,K,D,tempo_1no_s,desvio_1no_s,tempo_32nos_s,desvio_32nos_s,speedup_32_vs_1\n")
        for r in data:
            f.write(f"{r['key']},{r['n_pts']},{r['k']},{r['d']},{r['t1']:.4f},{r['sd1']:.4f},"
                    f"{r['t32']:.4f},{r['sd32']:.4f},{r['sp']:.3f}\n")


def table_png(data):
    header = ["Cenário", "N", "K", "D", "Tempo de loop\n1 nó (s)", "Tempo de loop\n32 nós (s)", "Speedup\n32 vs. 1 nó"]
    rows = [[r["name"], r["n"], str(r["k"]), str(r["d"]),
             f"{fmt(r['t1'], 2)} ± {fmt(r['sd1'], 2)}", f"{fmt(r['t32'], 2)} ± {fmt(r['sd32'], 2)}",
             f"{fmt(r['sp'], 1)}×"] for r in data]
    a1, a32 = ATUAL[1], ATUAL[32]
    rows.append(["Atual (artigo)", "3×10⁸", "50", "2", f"{fmt(a1[0], 2)} ± {fmt(a1[1], 2)}",
                 f"{fmt(a32[0], 2)} ± {fmt(a32[1], 2)}", f"{fmt(a1[0] / a32[0], 1)}×"])

    fig = plt.figure(figsize=(11.2, 4.0))
    ax = fig.add_axes([0.01, 0.17, 0.98, 0.72])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1],
                  colWidths=[0.2, 0.12, 0.06, 0.06, 0.19, 0.19, 0.14])
    tb.auto_set_font_size(False)
    tb.set_fontsize(12)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#9a9995")
        if i == 0:
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(weight="bold", color=INK)
        elif i == len(rows):  # referência do artigo
            cell.set_text_props(color=GRAY, style="italic")
        else:
            cell.set_text_props(color=INK, weight="bold" if j == 6 else "normal")
    fig.text(0.01, 0.93, "StarPU+MPI híbrido — tempo de loop e speedup com 32 nós (média de 3 execuções)",
             fontsize=13, color=INK)
    fig.text(0.01, 0.02, "30 iterações, AWS g6.4xlarge (6 workers de CPU + 1 GPU L4 por nó), 1024 chunks, "
             "escalonador dmda.\nTempo = rank mais lento de cada execução. Atual (artigo): mesma métrica, "
             "rodada de 28/10 com 2 calibrações + 3 medições.", fontsize=9.5, color=GRAY)
    fig.savefig(os.path.join(OUT, "novos_cenarios_tabela.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def scaling_chart(data):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": GRAY, "ytick.color": GRAY})
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    nodes = [1, 2, 4, 8, 16, 32]
    ax.plot(nodes, nodes, "--", color=GRAY, linewidth=1.5, label="Ideal (linear)")
    ax.annotate("Ideal 32×", (32, 32), xytext=(-6, 9), textcoords="offset points", ha="right", va="center",
                fontsize=9, color=GRAY)

    for r in data:
        ax.plot([1, 32], [1, r["sp"]], color=r["color"], linewidth=2, marker=r["marker"], markersize=7,
                markeredgecolor="white", markeredgewidth=1,
                label=f"{r['name']} (N={r['n']}, K={r['k']}, D={r['d']})")
        ax.annotate(f"{r['name'].split()[-1]} {fmt(r['sp'], 1)}×", (32, r["sp"]), xytext=(8, r["dy"]),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    ns = sorted(ATUAL)
    t1 = ATUAL[1][0]
    ax.plot(ns, [t1 / ATUAL[n][0] for n in ns], color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1, label="Atual do artigo (N=3×10⁸, K=50, D=2)")
    ax.annotate(f"Atual {fmt(t1 / ATUAL[32][0], 1)}×", (32, t1 / ATUAL[32][0]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(nodes, [str(n) for n in nodes])
    ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(0.85, 100)
    ax.set_xlabel("Número de nós")
    ax.set_ylabel("Speedup sobre 1 nó")
    ax.set_title("StarPU+MPI híbrido — escalabilidade de 1 para 32 nós", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.text(0.01, -0.02, "Cenários novos medidos com 1 e 32 nós (a linha liga as duas medições); "
             "o atual, com 1, 2, 4 e 32 nós. Média de 3 execuções.", fontsize=7.5, color=GRAY)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"novos_cenarios_escala.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load()
    write_csv(data)
    table_png(data)
    scaling_chart(data)
    for r in data:
        print(f"{r['key']:<5} 1 nó {r['t1']:7.2f} s | 32 nós {r['t32']:5.2f} s | speedup {r['sp']:5.2f}x")
