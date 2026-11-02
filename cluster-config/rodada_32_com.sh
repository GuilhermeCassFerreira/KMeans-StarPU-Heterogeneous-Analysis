#!/usr/bin/env bash
set -uo pipefail
# =============================================================================
# rodada_32_com.sh — Testes de comunicação do StarPU-MPI com 32 nós, antes dos testes finais.
# Mesmos chunks do diagnóstico (v2_diag32), que é a referência sem ajustes.
# Uso (no nó 0, da RAIZ do projeto, depois de cluster-mpi.sh set-ips com os 32 IPs):
#   bash cluster-config/rodada_32_com.sh [--dry-run]
#
# 1. Gera as entradas no nó 0 (completas e a do SIFT com os pontos de um nó).
# 2. Nó 0 sozinho: calibra o SIFT com 64 chunks por nó e 4 threads por ponto na GPU.
# 3. Confere em todos os nós que o HT está desligado (8 CPUs).
# 4. 32 nós, execução de conferência (não entra nas medidas): STARPU_DISPLAY_BINDINGS=1 mostra
#    a CPU de cada thread do StarPU; confere 8 threads, cada um numa CPU diferente, em cada nó.
# 5. 32 nós, configurações alternadas, 3 execuções de cada por cenário:
#      B = STARPU_MPI_NDETACHED_SEND=0 + STARPU_MPI_NREADY_PROCESS=64 (0 trava o StarPU 1.4.3)
#      C = B + KMEANS_PODAR=1
# 6. 32 nós: SIFT com 64 chunks por nó e 4 threads por ponto, com C, 3 execuções.
#
# Medição por nó ligada em todas as execuções com 32 nós. Cada chamada tem limite de tempo.
# No fim (ou ao abortar), grava results/experiments/v2_com32/ESTADO e agenda o desligamento
# de todos os nós para 15 min depois, como proteção caso o cluster não seja destruído antes.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUN="bash cluster-config/run-experiments.sh"
TOOL="cluster-config/cluster-mpi.sh"
BASE="v2_com32"
RES="results/experiments/${BASE}"
CALIB="--calib-hostname g6.4xlarge-v2 --calib-dir calibracao_aws/sampling_v2"
COMUM="-x KMEANS_ASSIGN_CALC_JUNTOS=0"
MED="-x KMEANS_MEDIR_NOS=1"
VAR_B="-x STARPU_MPI_NDETACHED_SEND=0 -x STARPU_MPI_NREADY_PROCESS=64"
VAR_C="${VAR_B} -x KMEANS_PODAR=1"
TPP4="-x KMEANS_ASSIGN_THREADS_PONTO=4"
LIMITE_PRIMEIRA=1800   # s; a primeira chamada com 32 nós sincroniza e compila em todos
LIMITE=600             # s; as outras têm 1 a 3 execuções de ~1 min
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

declare -A N_TOTAL=([pq]=102400000 [deep]=20480000 [sift]=10240000)
declare -A DIMS=([pq]=16 [deep]=96 [sift]=128)
declare -A CHUNKS32=([pq]=3200 [deep]=2048 [sift]=1024)

log() { printf '[com32 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

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

# rodar <sessão> <aborta se falhar: 1/0> <limite em s> <args do run-experiments...>
rodar() {
  local sessao="$1" critica="$2" limite="$3"; shift 3
  local cmd="$RUN $CALIB --session ${BASE}/${sessao} $*"
  log "Sessão ${sessao}"
  if ((DRY_RUN)); then echo "  $cmd"; return 0; fi
  eval "timeout -k 30 ${limite} $cmd" >> "${RES}/${sessao}.out" 2>&1
  local rc=$?
  if ((rc == 124 || rc == 137)); then
    log "AVISO: sessão ${sessao} passou de ${limite} s; matando o kmeans_starpu em todos os nós"
    pkill -9 -f mpirun 2>/dev/null
    "$TOOL" exec-all "pkill -9 -f kmeans_starpu" >/dev/null 2>&1 || true
  fi
  # Medição por nó: o rank 0 grava <saída>/nos.csv; as pastas de saída se repetem entre sessões
  local f
  for f in output_bench/exp_starpu_hybrid_np*_*/nos.csv; do
    [[ -f "$f" ]] || continue
    mkdir -p "${RES}/${sessao}"
    mv "$f" "${RES}/${sessao}/nos_$(basename "$(dirname "$f")")_$(date +%H%M%S).csv"
  done
  if ((rc != 0)) || ! sessao_ok "$sessao"; then
    ((critica)) && abortar "sessão ${sessao} falhou (ver ${RES}/${sessao}.out)"
    log "AVISO: sessão ${sessao} falhou"
    return 1
  fi
  return 0
}

inicio=$(date +%s)

# ---- 1. Entradas no nó 0 ----------------------------------------------------
for arq in "input_pq.bin:${N_TOTAL[pq]}:16" "input_deep.bin:${N_TOTAL[deep]}:96" \
           "input_sift.bin:${N_TOTAL[sift]}:128" "input_sift_32.bin:$((N_TOTAL[sift] / 32)):128"; do
  IFS=: read -r f n d <<< "$arq"
  log "Gerando ${f} (${n} pontos, D=${d})"
  ((DRY_RUN)) && continue
  "$TOOL" generate-input "$n" "$d" 42 "$f" 8 > "${RES}/gerar_${f}.out" 2>&1 \
    && [[ -s "$f" ]] || abortar "não gerou ${f}"
done

# ---- 2. Nó 0 sozinho: calibração do SIFT com 64 chunks e 4 threads por ponto ----
# A primeira chamada desliga o HT em todos os nós e compila só no nó 0
S64_OK=1
rodar cal_sift64_t4 0 1200 --nodes 1 --no-generate --env cluster-config/cenario_sift.env \
  --input input_sift_32.bin --n $((N_TOTAL[sift] / 32)) --chunks-total 64 --warmup-reps 3 --timed-reps 1 \
  --extra-mpirun-args "'$COMUM $TPP4'" || S64_OK=0
((S64_OK)) || log "SIFT com 64 chunks e 4 threads descartado: a calibração falhou"

# ---- 3. HT desligado em todos os nós ----------------------------------------
restaurar_ips
if ((!DRY_RUN)); then
  "$TOOL" exec-all 'echo "SMT=$(cat /sys/devices/system/cpu/smt/control) CPUS=$(nproc)"' \
    > "${RES}/smt.txt" 2>&1
  echo "hwloc-ps no nó 0: $(command -v hwloc-ps || echo ausente)" >> "${RES}/smt.txt"
  ok="$(grep -c '^SMT=off CPUS=8$' "${RES}/smt.txt")"
  log "HT desligado e 8 CPUs em ${ok}/32 nós"
  ((ok == 32)) || abortar "HT ligado ou número de CPUs diferente de 8 em algum nó (ver ${RES}/smt.txt)"
fi

# ---- 4. 32 nós: conferência da fixação das threads (não entra nas medidas) ----
# A primeira chamada com 32 nós sincroniza o código e a calibração e compila em todos
NOS32="--nodes 32 --no-generate --freeze-calibration --warmup-reps 0"
SEM_SETUP="--skip-setup --no-build --no-sync"
todo() { local c="$1"; echo "--env cluster-config/cenario_${c}.env --input input_${c}.bin --chunks-total ${CHUNKS32[$c]}"; }

rodar conferencia_fixacao 1 "$LIMITE_PRIMEIRA" $NOS32 $(todo pq) --timed-reps 1 \
  --extra-mpirun-args "'--tag-output $COMUM $VAR_B -x STARPU_DISPLAY_BINDINGS=1'"
if ((!DRY_RUN)); then
  # Linhas marcadas pelo --tag-output com o rank ("[1,R]<stdout>:"). Para cada rank, as threads
  # do kmeans_starpu com nome de worker (CPU n, CUDA n) ou MPI: precisam ser 8, com máscara
  # de uma CPU só e todas diferentes
  awk '
    match($0, /^\[[0-9]+,[0-9]+\]<std[a-z]+>:/) {
      r = substr($0, 1, RLENGTH); sub(/^\[[0-9]+,/, "", r); sub(/\].*/, "", r)
      l = substr($0, RLENGTH + 1); sub(/^ /, "", l); nf = split(l, f, "\t")
      if (l ~ /^== Binding for rank/) { ranks[r] = 1; next }
      if (l ~ /^[0-9]/) { dentro[r] = (f[nf] ~ /kmeans_starpu$/); next }
      if (l ~ /^ [0-9]/ && dentro[r] && f[nf] ~ /^(CPU [0-9]+|CUDA [0-9]+|MPI)$/) {
        m = tolower(f[2]); sub(/^0x0*/, "", m)
        n[r]++; if (m !~ /^[1248]0*$/) multi[r]++; if (mascara[r, m]++) rep[r]++
      }
    }
    END {
      for (r in ranks) {
        ok = (n[r] == 8 && !multi[r] && !rep[r])
        printf "rank %s: %d threads, %s\n", r, n[r], ok ? "uma CPU por thread, todas diferentes" : "FALHA"
        bons += ok
      }
      printf "TOTAL %d nós ok\n", bons
    }' "${RES}/conferencia_fixacao/logs/starpu_hybrid_np32_metricas1.log" | sort -V > "${RES}/fixacao.txt"
  log "Fixação: $(grep "^TOTAL" "${RES}/fixacao.txt") (detalhes em ${RES}/fixacao.txt)"
  grep -q '^TOTAL 32 ' "${RES}/fixacao.txt" || log "AVISO: fixação diferente do esperado em algum nó"
fi

# ---- 5. 32 nós: B e C alternadas, 3 execuções de cada por cenário ------------
declare -A VARS=([B]="$VAR_B" [C]="$VAR_C")
for r in 1 2 3; do
  ordem="B C"; ((r % 2 == 0)) && ordem="C B"
  for c in pq deep sift; do
    for cfg in $ordem; do
      rodar "${cfg}_${c}" 1 "$LIMITE" $NOS32 $SEM_SETUP $(todo "$c") --timed-reps 1 --rep-offset $((r - 1)) \
        --extra-mpirun-args "'$COMUM $MED ${VARS[$cfg]}'"
    done
  done
done

# ---- 6. 32 nós: SIFT com 64 chunks por nó e 4 threads por ponto, com C ----------
if ((S64_OK)); then
  rodar C_sift64_t4 0 "$LIMITE" $NOS32 $SEM_SETUP --env cluster-config/cenario_sift.env \
    --input input_sift.bin --chunks-total 2048 --timed-reps 3 \
    --extra-mpirun-args "'$COMUM $MED $VAR_C $TPP4'"
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
ESTADO="CONCLUIDO em $(( ($(date +%s) - inicio) / 60 )) min (SIFT 64 com 4 threads: $( ((S64_OK)) && echo testado || echo descartado))"
