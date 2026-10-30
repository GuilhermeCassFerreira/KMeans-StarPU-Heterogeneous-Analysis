"""Gráfico de Gantt por worker a partir do trace FxT do StarPU (tasks.rec do starpu_fxt_tool).

Uma linha por worker (GPU e CPUs) dos ranks rastreados, com as tarefas coloridas pelo kernel.
Gera <saida>.png/.pdf com a execução inteira e <saida>_zoom.png/.pdf com 2 iterações.
Uso: python3 scripts/plot_gantt_fxt.py <dir_traces> <saida_sem_extensao>
"""
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

KERNELS = [  # nome no trace -> rótulo, cor (ordem categórica fixa)
    ("kmeans_assign_model_mpi", "Assign", "#2a78d6"),
    ("kmeans_calculate_model_mpi", "Calculate", "#eb6834"),
    ("kmeans_accumulate_mpi", "Accumulate", "#1baf7a"),
    ("kmeans_update_model_mpi", "Update / Clean", "#eda100"),
    ("kmeans_clean_model_mpi", "Update / Clean", "#eda100"),
]
COLOR = {k: c for k, _, c in KERNELS}
LABEL = {k: l for k, l, _ in KERNELS}
GRAY, INK, GRID = "#52514e", "#0b0b0b", "#e6e5e1"


def load_tasks(path):
    tasks = []
    for rec in open(path).read().split("\n\n"):
        d = dict(l.split(": ", 1) for l in rec.strip().split("\n") if ": " in l)
        if d.get("Name") in COLOR and "StartTime" in d:
            tasks.append(dict(rank=int(d["MPIRank"]), worker=int(d["WorkerId"]), gpu=d["MemoryNode"] != "0",
                              name=d["Name"], start=float(d["StartTime"]), end=float(d["EndTime"])))
    return tasks


def load_centroid_arrivals(path):
    """Instantes em que os centroides chegam a cada rank: a maior mensagem enviada pelo rank 0 (K*D doubles)."""
    comms = []
    for rec in open(path).read().split("\n\n"):
        d = dict(l.split(": ", 1) for l in rec.strip().split("\n") if ": " in l)
        if d.get("Src") == "0" and "Size" in d:
            comms.append((int(d["Size"]), int(d["Dst"]), float(d["RecvTime"])))
    size = max(c[0] for c in comms)
    arrivals = defaultdict(list)
    for s, dst, t in comms:
        if s == size:
            arrivals[dst].append(t)
    return arrivals


def rows_for(tasks):
    """Linhas do gráfico: por rank, a GPU primeiro e depois as CPUs."""
    workers = sorted({(t["rank"], t["worker"], t["gpu"]) for t in tasks}, key=lambda x: (x[0], not x[2], x[1]))
    rows, cpu_idx = {}, defaultdict(int)
    for rank, w, gpu in workers:
        if gpu:
            label = f"Rank {rank} · GPU"
        else:
            cpu_idx[rank] += 1
            label = f"Rank {rank} · CPU {cpu_idx[rank]}"
        rows[(rank, w)] = label
    return rows


def draw(tasks, arrivals, t0, t1, out, title):
    rows = rows_for(tasks)
    order = list(rows)
    n = len(order)
    fig, ax = plt.subplots(figsize=(11, 0.42 * n + 1.6))
    for t in tasks:
        if t["end"] < t0 or t["start"] > t1:
            continue
        y = n - 1 - order.index((t["rank"], t["worker"]))
        ax.broken_barh([(t["start"] - t0, t["end"] - t["start"])], (y - 0.36, 0.72),
                       facecolors=COLOR[t["name"]], edgecolor="none")
    # Chegada dos centroides da iteração seguinte (comunicação vinda do rank 0)
    for rank, times in arrivals.items():
        gpu_rows = [i for i, k in enumerate(order) if k[0] == rank]
        if not gpu_rows:
            continue
        y_top = n - 1 - gpu_rows[0] + 0.5
        xs = [x - t0 for x in times if t0 <= x <= t1]
        ax.plot(xs, [y_top + 0.05] * len(xs), "v", color=INK, markersize=5, linestyle="none")
    # Separador entre ranks
    ranks = sorted({k[0] for k in order})
    for r in ranks[1:]:
        first = [i for i, k in enumerate(order) if k[0] == r][0]
        ax.axhline(n - first - 0.5, color=GRAY, linewidth=0.8)
    ax.set_yticks(range(n), [rows[k] for k in reversed(order)], fontsize=8.5, color=INK)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(-0.6, n + 0.2)
    ax.set_xlabel("Tempo (ms)", color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=c, label=l) for l, c in dict((l, c) for _, l, c in KERNELS).items()]
    handles.append(plt.Line2D([], [], marker="v", color=INK, linestyle="none", label="Centroides chegam do rank 0"))
    ax.legend(handles=handles, frameon=False, ncol=5, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.07))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    trace_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tasks = load_tasks(os.path.join(trace_dir, "tasks.rec"))
    arrivals = load_centroid_arrivals(os.path.join(trace_dir, "comms.rec"))
    t0 = min(t["start"] for t in tasks)
    t1 = max(t["end"] for t in tasks)
    draw(tasks, arrivals, t0, t1, out, "StarPU híbrido, 32 nós, SIFT — tarefas por worker (ranks 0 e 1)")
    # Zoom: 2 iterações no meio da execução (entre chegadas consecutivas de centroides no rank 1)
    arr = sorted(arrivals.get(1, []))
    if len(arr) >= 17:
        draw(tasks, arrivals, arr[14] - 5, arr[16] + 5, out + "_zoom",
             "Zoom em 2 iterações — o intervalo sem tarefas antes de cada ▼ é a espera pelos centroides")
    # Ocupação por worker na janela do loop
    busy = defaultdict(float)
    for t in tasks:
        busy[(t["rank"], t["worker"])] += t["end"] - t["start"]
    rows = rows_for(tasks)
    print(f"janela do loop: {t1 - t0:.1f} ms")
    for k in rows:
        print(f"{rows[k]:<18} ocupado {busy[k]:8.1f} ms ({100 * busy[k] / (t1 - t0):4.1f}%)")


if __name__ == "__main__":
    main()
