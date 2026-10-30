"""Estimativa de escalabilidade do StarPU+MPI híbrido para novos cenários.

Modelo ajustado às medições (1, 2, 4 e 32 nós): t(n) = a/n + c·n, onde a é a
computação (proporcional a N·K·D) e c o custo que cresce com o número de nós.
Gera results/figures/estimativa_escala.{png,pdf}.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figures")

# Ajuste aos tempos medidos do cenário atual (N=3e8, K=50, D=2): t(1)=15,431 s e t(32)=1,102 s
T1, T32 = 15.4307, 1.10165
C = (T32 - T1 / 32) / (32 - 1 / 32)
A_BASE = T1 - C
NKD_BASE = 3e8 * 50 * 2

MEASURED = {1: 15.4307, 2: 7.7648, 4: 3.9654, 32: 1.10165}
SCENARIOS = [  # nome, rótulo curto, N, K, D, cor, marcador, deslocamento do rótulo final (pt)
    ("Descritores Deep (N=2×10⁷, K=256, D=96)", "Deep", 2e7, 256, 96, "#1baf7a", "s", 2),
    ("Subvetores de PQ (N=10⁸, K=256, D=16)", "PQ", 1e8, 256, 16, "#eb6834", "^", -10),
    ("Descritores SIFT (N=10⁷, K=256, D=128)", "SIFT", 1e7, 256, 128, "#eda100", "D", -22),
]
BLUE, GRAY, INK = "#2a78d6", "#52514e", "#0b0b0b"
NODES = [1, 2, 4, 8, 16, 32]


def speedup(a, n):
    return (a + C) / (a / n + C * n)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "xtick.color": GRAY, "ytick.color": GRAY})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.plot(NODES, NODES, "--", color=GRAY, linewidth=1.5, label="Ideal (linear)")
    ax.annotate("Ideal 32×", (32, 32), xytext=(8, 10), textcoords="offset points", va="center", fontsize=9, color=GRAY)

    for name, short, n_pts, k, d, color, marker, dy in SCENARIOS:
        a = A_BASE * (n_pts * k * d) / NKD_BASE
        s = [speedup(a, n) for n in NODES]
        ax.plot(NODES, s, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1, label=name + " — estimado")
        ax.annotate(f"{short} {s[-1]:.1f}×".replace(".", ","), (32, s[-1]), xytext=(8, dy), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    # Cenário atual: pontos medidos + curva do modelo
    model = [speedup(A_BASE, n) for n in NODES]
    ax.plot(NODES, model, ":", color=BLUE, linewidth=1.5)
    ns = sorted(MEASURED)
    ax.plot(ns, [MEASURED[1] / MEASURED[n] for n in ns], color=BLUE, linewidth=0, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1, label="Atual (N=3×10⁸, K=50, D=2) — medido")
    ax.annotate("14,0× (medido)", (32, MEASURED[1] / MEASURED[32]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(NODES, [str(n) for n in NODES])
    ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(0.85, 110)
    ax.set_xlabel("Número de nós")
    ax.set_ylabel("Speedup sobre 1 nó")
    ax.set_title("StarPU+MPI híbrido — escalabilidade medida e estimada", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.text(0.01, -0.02, "Estimativa: t(n) = a/n + c·n, ajustado às medições do cenário atual (1, 2, 4 e 32 nós); "
             "a proporcional a N·K·D. A linha pontilhada azul é o modelo do cenário atual.",
             fontsize=7.5, color=GRAY)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"estimativa_escala.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
