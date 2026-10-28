"""Tabela e gráfico de escalabilidade do StarPU+MPI híbrido (1, 2, 4 e 32 nós).

Lê o "Tempo do loop" das execuções de medição (rank mais lento, como na Tabela 4
do artigo) e gera, em results/figures/:
  starpu_hibrido_escala.csv / .tex / .pdf / .png
"""
import glob
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figures")
T_SEQ = 659.0  # tempo de loop da versão sequencial (s), Tabela 3

# Logs das 3 execuções de medição de cada configuração
RUNS = {
    1: "logs/starpu_np1/starpu_hybrid/*_metricas.log",
    2: "results/experiments/starpu_np2_hybrid/logs/starpu_hybrid_np2_metricas*.log",
    4: "results/experiments/starpu_np4/logs/starpu_hybrid_np4_metricas*.log",
    32: "results/experiments/starpu_np32_hybrid/logs/starpu_hybrid_np32_metricas*.log",
}
# 1 nó: "Tempo do loop : X ms"; vários nós: "Tempo do loop (max) : X ms [min: Y]"
LOOP_RE = re.compile(r"Tempo do loop(?: \(max\))?\s*:\s*([\d.]+)\s*ms")

BLUE = "#2a78d6"   # série medida
GRAY = "#52514e"   # referência (ideal)
INK = "#0b0b0b"
MUTED = "#8a8984"


def fmt(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def load():
    rows = []
    for nodes, pattern in RUNS.items():
        times = []
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            with open(path, errors="replace") as f:
                m = LOOP_RE.search(f.read())
            if m:
                times.append(float(m.group(1)) / 1000.0)
        if len(times) != 3:
            raise SystemExit(f"{nodes} nó(s): esperava 3 medições, achei {len(times)} em {pattern}")
        rows.append((nodes, statistics.mean(times), statistics.stdev(times)))
    t1 = rows[0][1]
    return [
        dict(nodes=n, t=t, sd=sd, sp_seq=T_SEQ / t, sp_1=t1 / t, eff=t1 / t / n)
        for n, t, sd in rows
    ]


def write_tables(data):
    with open(os.path.join(OUT, "starpu_hibrido_escala.csv"), "w") as f:
        f.write("nos,tempo_loop_s,desvio_s,speedup_vs_sequencial,speedup_vs_1no,eficiencia\n")
        for d in data:
            f.write(f"{d['nodes']},{d['t']:.6f},{d['sd']:.6f},{d['sp_seq']:.4f},{d['sp_1']:.4f},{d['eff']:.4f}\n")

    with open(os.path.join(OUT, "starpu_hibrido_escala.tex"), "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Tempo de \\textit{loop} e \\textit{speedup} do StarPU+MPI híbrido.}\n")
        f.write("\\label{tab:starpu-escala}\n\\begin{tabular}{rrrrr}\n\\hline\n")
        f.write("Nós & Tempo de \\textit{loop} (s) & \\textit{Speedup} vs. sequencial "
                "& \\textit{Speedup} vs. 1 nó & Eficiência \\\\\n\\hline\n")
        for d in data:
            f.write(f"{d['nodes']} & {fmt(d['t'], 3)} $\\pm$ {fmt(d['sd'], 3)} & {fmt(d['sp_seq'], 1)}$\\times$ "
                    f"& {fmt(d['sp_1'], 2)}$\\times$ & {fmt(d['eff'] * 100, 0)}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def plot(data):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    nodes = [d["nodes"] for d in data]
    labels = [str(n) for n in nodes]

    # (a) Tempo de loop: barras (magnitude), com desvio padrão
    ax1.bar(labels, [d["t"] for d in data], yerr=[d["sd"] for d in data], color=BLUE,
            width=0.6, capsize=3, error_kw={"ecolor": GRAY, "elinewidth": 1})
    for i, d in enumerate(data):
        ax1.text(i, d["t"] + 0.3, fmt(d["t"], 2) + " s", ha="center", va="bottom", fontsize=9, color=INK)
    ax1.set_xlabel("Número de nós")
    ax1.set_ylabel("Tempo de loop (s)")
    ax1.set_title("(a) Tempo de loop", loc="left", fontsize=10, color=INK)

    # (b) Speedup sobre 1 nó vs. ideal, escala log2 nos dois eixos
    ax2.plot(nodes, nodes, linestyle="--", color=GRAY, linewidth=1.5, label="Ideal (linear)")
    ax2.plot(nodes, [d["sp_1"] for d in data], color=BLUE, linewidth=2, marker="o",
             markersize=7, markeredgecolor="white", markeredgewidth=1.5, label="StarPU+MPI híbrido")
    last = data[-1]
    ax2.annotate(f"{fmt(last['sp_1'], 1)}×\n({fmt(last['eff'] * 100, 0)}% do ideal)", (last["nodes"], last["sp_1"]),
                 xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)
    ax2.annotate("32×", (32, 32), xytext=(8, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=9, color=GRAY)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlim(0.8, 150)  # espaço à direita para os rótulos dos pontos finais
    ax2.set_xticks(nodes, labels)
    ax2.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax2.set_xlabel("Número de nós")
    ax2.set_ylabel("Speedup sobre 1 nó")
    ax2.set_title("(b) Speedup", loc="left", fontsize=10, color=INK)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"starpu_hibrido_escala.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load()
    write_tables(data)
    plot(data)
    for d in data:
        print(f"{d['nodes']:>2} nós: {d['t']:7.3f} s ± {d['sd']:.3f} | vs seq {d['sp_seq']:6.1f}x | "
              f"vs 1 nó {d['sp_1']:5.2f}x | eficiência {d['eff'] * 100:4.0f}%")
