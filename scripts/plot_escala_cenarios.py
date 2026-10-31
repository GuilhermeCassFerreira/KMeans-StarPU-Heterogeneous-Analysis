"""Tabela e gráfico de escalabilidade do StarPU+MPI híbrido nos cenários PQ, Deep e SIFT, de 1 a 32 nós.

Lê o "Tempo do loop" (rank mais lento) das 5 medições de cada configuração em
results/experiments/escala/<cenario>_escala_np<N>/results.csv (todas no mesmo cluster, com a
calibração congelada) e gera, em results/figures/:
  escala_cenarios.csv, escala_cenarios_tabela.png, escala_cenarios.png/.pdf
O cenário do artigo (N=3e8, K=50, D=2; 1, 2, 4 e 32 nós, outra rodada) entra no gráfico como referência.
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "experiments", "escala")
OUT = os.path.join(ROOT, "results", "figures")
NODES = [1, 2, 4, 8, 16, 32]

SCENARIOS = [  # chave, nome, N formatado, D, cor, marcador, deslocamento do rótulo final (pt)
    ("pq", "PQ", "1,024×10⁸", 16, "#eb6834", "^", -11),
    ("deep", "Deep", "2,048×10⁷", 96, "#1baf7a", "s", 0),
    ("sift", "SIFT", "1,024×10⁷", 128, "#eda100", "D", 11),
]
# Cenário do artigo (results/figures/starpu_hibrido_escala.csv)
ATUAL = {1: 15.4307, 2: 7.7648, 4: 3.9654, 32: 1.10165}
BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#8a8984", "#e6e5e1"


def fmt(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def loop_times(key, nodes):
    path = os.path.join(EXP, f"{key}_escala_np{nodes}", "results.csv")
    rows = [r for r in csv.DictReader(open(path)) if r["rep_kind"] == "metricas" and r["run_ok"] == "1"]
    if len(rows) != 5:
        raise SystemExit(f"{path}: esperava 5 medições, achei {len(rows)}")
    return [float(r["loop_ms"]) / 1000 for r in rows]


def load():
    data = {}
    for key, *_ in SCENARIOS:
        t1 = statistics.mean(loop_times(key, 1))
        for n in NODES:
            t = loop_times(key, n)
            m = statistics.mean(t)
            data[key, n] = dict(mean=m, sd=statistics.stdev(t), sp=t1 / m)
    return data


def write_csv(data):
    with open(os.path.join(OUT, "escala_cenarios.csv"), "w") as f:
        f.write("cenario,nos,tempo_loop_s,desvio_s,speedup_vs_1no\n")
        for key, *_ in SCENARIOS:
            for n in NODES:
                d = data[key, n]
                f.write(f"{key},{n},{d['mean']:.4f},{d['sd']:.4f},{d['sp']:.3f}\n")


def table_png(data):
    header = ["Nós"]
    for _, name, *_ in SCENARIOS:
        header += [f"{name}\ntempo (s)", f"{name}\nspeedup"]
    rows = []
    for n in NODES:
        row = [str(n)]
        for key, *_ in SCENARIOS:
            d = data[key, n]
            row += [f"{fmt(d['mean'], 2)} ± {fmt(d['sd'], 2)}", f"{fmt(d['sp'], 1)}×"]
        rows.append(row)

    fig = plt.figure(figsize=(11.6, 4.3))
    ax = fig.add_axes([0.01, 0.17, 0.98, 0.72])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1],
                  colWidths=[0.07] + [0.19, 0.12] * 3)
    tb.auto_set_font_size(False)
    tb.set_fontsize(11.5)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#9a9995")
        if i == 0:
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK, weight="bold" if (j > 0 and j % 2 == 0 and i == len(rows)) else "normal")
    fig.text(0.01, 0.93, "StarPU+MPI híbrido — tempo de loop e speedup de 1 a 32 nós (média de 5 execuções)",
             fontsize=13, color=INK)
    fig.text(0.01, 0.02, "K = 256, 30 iterações, AWS g6.4xlarge (6 workers de CPU + 1 GPU L4 por nó), 1024 chunks, "
             "dmda, calibração reinstalada antes de cada execução.\nPQ: N = 1,024×10⁸, D = 16; Deep: N = 2,048×10⁷, "
             "D = 96; SIFT: N = 1,024×10⁷, D = 128. Tempo = rank mais lento; todas as medições no mesmo cluster.",
             fontsize=9, color=GRAY)
    fig.savefig(os.path.join(OUT, "escala_cenarios_tabela.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def chart(data):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": GRAY, "ytick.color": GRAY})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) Tempo de loop × nós (log-log), com desvio padrão
    for key, name, n_txt, d, color, marker, _ in SCENARIOS:
        ax1.errorbar(NODES, [data[key, n]["mean"] for n in NODES], yerr=[data[key, n]["sd"] for n in NODES],
                     color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor="white",
                     markeredgewidth=1, capsize=2, label=f"{name} (N={n_txt}, D={d})")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(NODES, [str(n) for n in NODES])
    ax1.set_xlabel("Número de nós")
    ax1.set_ylabel("Tempo de loop (s)")
    ax1.set_title("(a) Tempo de loop", loc="left", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # (b) Speedup × nós, com o ideal e o cenário do artigo como referência
    ax2.plot(NODES, NODES, "--", color=GRAY, linewidth=1.5, label="Ideal (linear)")
    ax2.annotate("Ideal 32×", (32, 32), xytext=(-6, 9), textcoords="offset points", ha="right", va="center",
                 fontsize=9, color=GRAY)
    for key, name, n_txt, d, color, marker, dy in SCENARIOS:
        sp = [data[key, n]["sp"] for n in NODES]
        ax2.plot(NODES, sp, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor="white",
                 markeredgewidth=1, label=name)
        ax2.annotate(f"{name} {fmt(sp[-1], 1)}×", (32, sp[-1]), xytext=(8, dy), textcoords="offset points",
                     va="center", fontsize=9, color=INK)
    ns = sorted(ATUAL)
    sp_atual = [ATUAL[1] / ATUAL[n] for n in ns]
    ax2.plot(ns, sp_atual, color=BLUE, linewidth=1.5, linestyle=":", marker="o", markersize=5,
             markeredgecolor="white", markeredgewidth=1, label="Atual do artigo (K=50, D=2)")
    ax2.annotate(f"Atual {fmt(sp_atual[-1], 1)}×", (32, sp_atual[-1]), xytext=(8, 0), textcoords="offset points",
                 va="center", fontsize=9, color=INK)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(NODES, [str(n) for n in NODES])
    ax2.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax2.set_xlim(0.85, 100)
    ax2.set_xlabel("Número de nós")
    ax2.set_ylabel("Speedup sobre 1 nó")
    ax2.set_title("(b) Speedup", loc="left", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(0.01, -0.02, "StarPU+MPI híbrido, média de 5 execuções por configuração (barras: desvio padrão). "
             "O cenário do artigo é de outra rodada e serve só de referência.", fontsize=7.5, color=GRAY)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"escala_cenarios.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load()
    write_csv(data)
    table_png(data)
    chart(data)
    for key, name, *_ in SCENARIOS:
        print(name.ljust(5), "  ".join(f"{n}:{data[key, n]['mean']:.2f}s/{data[key, n]['sp']:.1f}x" for n in NODES))
