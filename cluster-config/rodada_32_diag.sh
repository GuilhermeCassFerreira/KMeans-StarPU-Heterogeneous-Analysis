#!/usr/bin/env bash
set -uo pipefail
# =============================================================================
# rodada_32_diag.sh — Diagnóstico da espera com 32 nós e teste do kernel com a distância
# expandida. Não gera números do artigo (medição por nó ligada em todas as execuções).
# Uso (no nó 0, da RAIZ do projeto, depois de cluster-mpi.sh set-ips com os 32 IPs):
#   bash cluster-config/rodada_32_diag.sh [--dry-run]
#
# 1. Gera as entradas no nó 0: completas e com os pontos de um nó (N/32).
# 2. Nó 0 sozinho, com N/32: calibra os tamanhos novos (PQ 100 por nó, SIFT 16 + 18),
#    mede o kernel normal e calibra/mede o kernel expandido nas configurações escolhidas.
#    Se o SSE do expandido diferir mais de 0,1% do normal, ele não roda com 32 nós.
# 3. 32 nós, calibração congelada: 13 execuções de diagnóstico (chunks, rank 0, escalonador).
# 4. 32 nós: kernel expandido, 3 execuções em cada cenário.
#
# Aborta se a compilação ou uma execução do kernel normal falhar. No fim (ou ao abortar),
# grava results/experiments/v2_diag32/ESTADO e agenda o desligamento de todos os nós para
# 15 min depois, como proteção caso o cluster não seja destruído antes.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUN="bash cluster-config/run-experiments.sh"
TOOL="cluster-config/cluster-mpi.sh"
BASE="v2_diag32"
RES="results/experiments/${BASE}"
CALIB="--calib-hostname g6.4xlarge-v2 --calib-dir calibracao_aws/sampling_v2"
COMUM="-x KMEANS_ASSIGN_CALC_JUNTOS=0"
MISTO_PQ="-x KMEANS_CHUNKS_PEQUENOS=60 -x KMEANS_FRACAO_PEQUENOS=0.2"
MISTO_SIFT="-x KMEANS_CHUNKS_PEQUENOS=18 -x KMEANS_FRACAO_PEQUENOS=0.3"
EXP="-x KMEANS_ASSIGN_EXPANDIDO=1"
LIMITE_SSE=0.001
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

declare -A N_TOTAL=([pq]=102400000 [deep]=20480000 [sift]=10240000)
declare -A DIMS=([pq]=16 [deep]=96 [sift]=128)

log() { printf '[diag32 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$RES" cluster-config/.cluster
cp cluster-config/.cluster/control_ips.txt "$RES/control_ips_32.txt"
cp cluster-config/.cluster/mpi_ips.txt "$RES/mpi_ips_32.txt" 2>/dev/null || true
restaurar_ips() {
  cp "$RES/control_ips_32.txt" cluster-config/.cluster/control_ips.txt
  [[ -f "$RES/mpi_ips_32.txt" ]] && cp "$RES/mpi_ips_32.txt" cluster-config/.cluster/mpi_ips.txt
}

ESTADO="INTERROMPIDO"
fim() {
  ((DRY_RUN)) && return
  echo "$ESTADO" > "$RES/ESTADO"
  log "Estado final: ${ESTADO}"
  restaurar_ips
  log "Agendando o desligamento de todos os nós para daqui a 15 min (proteção)"
  "$TOOL" exec-all "sudo shutdown -h +15" >/dev/null 2>&1 || true
}
trap fim EXIT
abortar() { ESTADO="ABORTADO: $*"; log "$ESTADO"; exit 1; }

# 0 se todas as execuções da sessão terminaram (run_ok = 1)
sessao_ok() {
  local csv="${RES}/$1/results.csv"
  [[ -f "$csv" ]] || return 1
  awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
           { n++; if ($h["run_ok"] != 1) bad = 1 }
           END { exit (n == 0 || bad) ? 1 : 0 }' "$csv"
}

# rodar <sessão> <aborta se falhar: 1/0> <args do run-experiments...>
rodar() {
  local sessao="$1" critica="$2"; shift 2
  local cmd="$RUN $CALIB --session ${BASE}/${sessao} $*"
  log "Sessão ${sessao}"
  if ((DRY_RUN)); then echo "  $cmd"; return 0; fi
  eval "$cmd" > "${RES}/${sessao}.out" 2>&1
  # Medição por nó: o rank 0 grava <saída>/nos.csv; as pastas de saída se repetem entre sessões
  local f
  for f in output_bench/exp_starpu_hybrid_np*_*/nos.csv; do
    [[ -f "$f" ]] || continue
    mv "$f" "${RES}/${sessao}/nos_$(basename "$(dirname "$f")").csv"
  done
  if ! sessao_ok "$sessao"; then
    ((critica)) && abortar "sessão ${sessao} falhou (ver ${RES}/${sessao}.out)"
    log "AVISO: sessão ${sessao} falhou"
    return 1
  fi
  return 0
}

# Média do SSE das medições da sessão
sse_medio() {
  awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
           $h["rep_kind"] == "metricas" && $h["run_ok"] == 1 { s += $h["sse"]; n++ }
           END { if (n) printf "%.10g", s / n }' "${RES}/$1/results.csv" 2>/dev/null
}

inicio=$(date +%s)

# ---- 1. Entradas no nó 0 ----------------------------------------------------
for c in pq deep sift; do
  for arq in "input_${c}.bin:${N_TOTAL[$c]}" "input_${c}_32.bin:$((N_TOTAL[$c] / 32))"; do
    f="${arq%%:*}"; n="${arq##*:}"
    log "Gerando ${f} (${n} pontos, D=${DIMS[$c]})"
    ((DRY_RUN)) && continue
    "$TOOL" generate-input "$n" "${DIMS[$c]}" 42 "$f" 8 > "${RES}/gerar_${f}.out" 2>&1 \
      && [[ -s "$f" ]] || abortar "não gerou ${f}"
  done
done

# ---- 2. Nó 0 sozinho, com os pontos de um nó --------------------------------
# A primeira chamada desliga o HT em todos os nós e compila só no nó 0
NO0="--nodes 1 --no-generate"
umno() { local c="$1"; echo "--env cluster-config/cenario_${c}.env --input input_${c}_32.bin --n $((N_TOTAL[$c] / 32))"; }

rodar n_pq_c100 1 $NO0 $(umno pq) --chunks-total 100 --warmup-reps 3 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM'"
SEM_SETUP="--skip-setup --no-build --no-sync"
rodar n_sift_g16p18 1 $NO0 $SEM_SETUP $(umno sift) --chunks-total 16 --warmup-reps 3 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM $MISTO_SIFT'"
rodar n_pq_g20p60 1 $NO0 $SEM_SETUP $(umno pq) --chunks-total 20 --warmup-reps 0 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM $MISTO_PQ'"
rodar n_deep_c64 1 $NO0 $SEM_SETUP $(umno deep) --chunks-total 64 --warmup-reps 0 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM'"
rodar n_sift_c32 1 $NO0 $SEM_SETUP $(umno sift) --chunks-total 32 --warmup-reps 0 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM'"

# Kernel expandido: calibração própria (modelo com outro nome) e 3 medições
EXP_OK=1
rodar x_pq_g20p60 0 $NO0 $SEM_SETUP $(umno pq) --chunks-total 20 --warmup-reps 3 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM $MISTO_PQ $EXP'" || EXP_OK=0
rodar x_deep_c64 0 $NO0 $SEM_SETUP $(umno deep) --chunks-total 64 --warmup-reps 3 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM $EXP'" || EXP_OK=0
rodar x_sift_c32 0 $NO0 $SEM_SETUP $(umno sift) --chunks-total 32 --warmup-reps 3 --timed-reps 3 \
  --extra-mpirun-args "'$COMUM $EXP'" || EXP_OK=0
if ((!DRY_RUN && EXP_OK)); then
  for p in pq_g20p60 deep_c64 sift_c32; do
    a="$(sse_medio "n_${p}")"; b="$(sse_medio "x_${p}")"
    dif="$(awk -v a="$a" -v b="$b" 'BEGIN { if (a == 0) print 1; else { d = (b - a) / a; print (d < 0 ? -d : d) } }')"
    log "SSE ${p}: normal ${a} | expandido ${b} | diferença relativa ${dif}"
    awk -v d="$dif" -v l="$LIMITE_SSE" 'BEGIN { exit !(d > l) }' && EXP_OK=0
  done
fi
((EXP_OK)) || log "Kernel expandido descartado: não roda com 32 nós"

# ---- 3. 32 nós: diagnóstico, calibração congelada -------------------------
# A primeira chamada sincroniza o código e a calibração e compila nos 32 nós
restaurar_ips
NOS32="--nodes 32 --no-generate --freeze-calibration --warmup-reps 0"
MED="-x KMEANS_MEDIR_NOS=1"
todo() { local c="$1"; echo "--env cluster-config/cenario_${c}.env --input input_${c}.bin"; }

rodar d01_pq_c32 1 $NOS32 $(todo pq) --chunks-total 1024 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED'"
rodar d02_pq_g20p60 1 $NOS32 $SEM_SETUP $(todo pq) --chunks-total 640 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED $MISTO_PQ'"
rodar d03_pq_c100 1 $NOS32 $SEM_SETUP $(todo pq) --chunks-total 3200 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED'"
rodar d04_pq_g20p60_r0 1 $NOS32 $SEM_SETUP $(todo pq) --chunks-total 640 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED $MISTO_PQ -x KMEANS_RANK0_MENOS_PEQUENOS=20'"
rodar d05_pq_g20p60_dmdas 1 $NOS32 $SEM_SETUP $(todo pq) --chunks-total 640 --timed-reps 1 --sched dmdas \
  --extra-mpirun-args "'$COMUM $MED $MISTO_PQ -x KMEANS_PRIORIDADE_GRANDES=1'"
rodar d06_pq_g20p60_dmdar 1 $NOS32 $SEM_SETUP $(todo pq) --chunks-total 640 --timed-reps 1 --sched dmdar \
  --extra-mpirun-args "'$COMUM $MED $MISTO_PQ'"
rodar d07_deep_c64 1 $NOS32 $SEM_SETUP $(todo deep) --chunks-total 2048 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED'"
rodar d08_sift_c32 1 $NOS32 $SEM_SETUP $(todo sift) --chunks-total 1024 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED'"
rodar d09_sift_g16p18 1 $NOS32 $SEM_SETUP $(todo sift) --chunks-total 512 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $MED $MISTO_SIFT'"
rodar d10_deep_c64_dmdas 1 $NOS32 $SEM_SETUP $(todo deep) --chunks-total 2048 --timed-reps 1 --sched dmdas \
  --extra-mpirun-args "'$COMUM $MED -x KMEANS_PRIORIDADE_GRANDES=1'"
rodar d11_deep_c64_dmdar 1 $NOS32 $SEM_SETUP $(todo deep) --chunks-total 2048 --timed-reps 1 --sched dmdar \
  --extra-mpirun-args "'$COMUM $MED'"
rodar d12_sift_c32_dmdas 1 $NOS32 $SEM_SETUP $(todo sift) --chunks-total 1024 --timed-reps 1 --sched dmdas \
  --extra-mpirun-args "'$COMUM $MED -x KMEANS_PRIORIDADE_GRANDES=1'"
rodar d13_sift_c32_dmdar 1 $NOS32 $SEM_SETUP $(todo sift) --chunks-total 1024 --timed-reps 1 --sched dmdar \
  --extra-mpirun-args "'$COMUM $MED'"

# ---- 4. 32 nós: kernel expandido -------------------------------------------
if ((EXP_OK)); then
  rodar e_pq_g20p60 0 $NOS32 $SEM_SETUP $(todo pq) --chunks-total 640 --timed-reps 3 \
    --extra-mpirun-args "'$COMUM $MED $MISTO_PQ $EXP'"
  rodar e_deep_c64 0 $NOS32 $SEM_SETUP $(todo deep) --chunks-total 2048 --timed-reps 3 \
    --extra-mpirun-args "'$COMUM $MED $EXP'"
  rodar e_sift_c32 0 $NOS32 $SEM_SETUP $(todo sift) --chunks-total 1024 --timed-reps 3 \
    --extra-mpirun-args "'$COMUM $MED $EXP'"
fi

((DRY_RUN)) && exit 0
# Resumo: tempo de loop, SSE e execuções ok de cada sessão
for d in "$RES"/*/; do
  s="$(basename "$d")"
  [[ -f "$d/results.csv" ]] || continue
  awk -F, -v s="$s" 'NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
       $h["rep_kind"] == "metricas" { n++; ok += ($h["run_ok"] == 1); t += $h["loop_ms"]; sse = $h["sse"] }
       END { if (n) printf "%-24s n=%d ok=%d loop=%.1f ms sse=%s\n", s, n, ok, t / n, sse }' "$d/results.csv"
done | tee "$RES/resumo.txt"
ESTADO="CONCLUIDO em $(( ($(date +%s) - inicio) / 60 )) min (kernel expandido: $( ((EXP_OK)) && echo testado || echo descartado))"
