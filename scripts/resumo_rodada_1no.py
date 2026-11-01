"""Resumo da rodada de 1 nó (cluster-config/rodada_1no_v2.sh).

Uso: python3 scripts/resumo_rodada_1no.py results/experiments/v2_1no

Para cada sessão: tempo de loop das medições (média, desvio e variação entre a maior e a
menor), parte do Assign e do Calculate na GPU, avisos de calibração e as opções usadas.
No teste de chunks, a diferença de cada configuração para os 32 chunks iguais do mesmo
cenário e do mesmo modo (Assign e Calculate separados ou juntos).
"""
import csv
import os
import statistics
import sys

base = sys.argv[1]


def ler(sessao):
    rows = [r for r in csv.DictReader(open(os.path.join(base, sessao, "results.csv")))
            if r["rep_kind"] == "metricas"]
    ok = [r for r in rows if r["run_ok"] == "1" and r["loop_ms"]]
    t = [float(r["loop_ms"]) / 1000 for r in ok]

    def gpu(k):
        c = sum(int(r[f"{k}_cpu"] or 0) for r in ok)
        g = sum(int(r[f"{k}_gpu"] or 0) for r in ok)
        return 100 * g / (c + g) if c + g else float("nan")

    return dict(n=len(t), falhas=len(rows) - len(ok), media=statistics.mean(t) if t else float("nan"),
                desvio=statistics.stdev(t) if len(t) > 1 else 0.0,
                var=(max(t) / min(t) - 1) * 100 if t else float("nan"),
                assign_gpu=gpu("assign"), calc_gpu=gpu("calculate"),
                avisos=sum(int(r["calib_warn"] or 0) for r in rows),
                opcoes=ok[0].get("opcoes", "") if ok else "")


sessoes = sorted(d for d in os.listdir(base)
                 if os.path.isfile(os.path.join(base, d, "results.csv")) and not d.endswith("_descartada"))
dados = {s: ler(s) for s in sessoes}
descartadas = sorted(d for d in os.listdir(base) if d.endswith("_descartada"))

cab = f"{'sessao':28s} {'n':>2s} {'loop (s)':>16s} {'var':>6s} {'Assign GPU':>10s} {'Calc GPU':>9s} {'avisos':>6s}"
print("== Calibração (todos os pontos, 1024 chunks)")
print(cab)
for s in sessoes:
    if s.startswith("calib_"):
        d = dados[s]
        print(f"{s:28s} {d['n']:2d} {d['media']:8.3f} ± {d['desvio']:5.3f} {d['var']:5.1f}% "
              f"{d['assign_gpu']:9.1f}% {d['calc_gpu']:8.1f}% {d['avisos']:6d}")

print("\n== Teste de chunks (pontos de um nó do cluster de 32); dif = em relação aos 32 iguais, mesmo modo")
print(cab + f" {'dif':>7s}  opcoes")
for s in sessoes:
    if not s.startswith("chunks_"):
        continue
    _, cen, nome, modo = s.split("_")
    d = dados[s]
    ref = dados.get(f"chunks_{cen}_c32_{modo}")
    dif = (d["media"] / ref["media"] - 1) * 100 if ref else float("nan")
    print(f"{s:28s} {d['n']:2d} {d['media']:8.3f} ± {d['desvio']:5.3f} {d['var']:5.1f}% "
          f"{d['assign_gpu']:9.1f}% {d['calc_gpu']:8.1f}% {d['avisos']:6d} {dif:+6.1f}%  {d['opcoes']}")

if descartadas:
    print("\nSessões repetidas (a primeira tentativa ficou em *_descartada):", ", ".join(descartadas))
falhas = [s for s, d in dados.items() if d["falhas"]]
if falhas:
    print("Sessões com execuções que falharam:", ", ".join(falhas))
