#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# rodada_1no_v2.sh — Rodada de 1 nó da versão nova do StarPU (uma g6.4xlarge)
# Uso (na instância, da RAIZ do projeto, depois de cluster-mpi.sh set-ips 10.0.0.10):
#   bash cluster-config/rodada_1no_v2.sh [--dry-run] [--so-calibracao | --so-chunks]
#
# 1. Calibração: cada cenário com todos os pontos e 1024 chunks, 2 calibrações + 2 medições.
#    Serve para conferir a calibração antes de usá-la no cluster; não vai para o artigo.
# 2. Teste de chunks: os pontos de um nó do cluster de 32 (N/32) em 1 processo, cada
#    configuração com Assign e Calculate separados e juntos, 3 calibrações + 3 medições.
#
# A calibração fica com nome de máquina próprio (g6.4xlarge-v2) e pasta própria, sem
# misturar com a de 30/10. Os 32 chunks iguais de N/32 têm o mesmo tamanho dos chunks das
# execuções com 1024 chunks, então reaproveitam a calibração da etapa 1.
#
# Se uma medição tiver aviso de calibração, falhar ou variar mais que LIMITE_VARIACAO, a
# sessão é guardada como _descartada e repetida uma vez (1 calibração a mais + medições).
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUN="bash cluster-config/run-experiments.sh"
BASE="v2_1no"                     # sessões em results/experiments/v2_1no/<nome>
CALIB="--calib-hostname g6.4xlarge-v2 --calib-dir calibracao_aws/sampling_v2"
LIMITE_VARIACAO=1.03              # maior / menor tempo de loop entre as medições
DRY_RUN=0
FAZER_CALIBRACAO=1
FAZER_CHUNKS=1

# cenario nome chunks(grandes) pequenos fracao — pequenos sempre com ~10 mil pontos ou mais
CONFIGS=(
  "pq   c32     32  0  0"
  "pq   g20p60  20  60 0.2"
  "pq   g20p30  20  30 0.1"
  "deep c32     32  0  0"
  "deep c25     25  0  0"
  "deep c64     64  0  0"
  "deep g16p16  16  16 0.25"
  "sift c32     32  0  0"
  "sift c20     20  0  0"
  "sift g12p8   12  8  0.25"
)

log() { printf '[rodada %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

while (($# > 0)); do
  case "$1" in
    --dry-run)       DRY_RUN=1;          shift ;;
    --so-calibracao) FAZER_CHUNKS=0;     shift ;;
    --so-chunks)     FAZER_CALIBRACAO=0; shift ;;
    *) echo "Argumento desconhecido: $1" >&2; exit 1 ;;
  esac
done

# Primeira chamada: desliga HT, sincroniza, compila e confere o compilador; as outras não
PRIMEIRA=1

# 0 se as medições da sessão estão boas: todas ok, sem aviso de calibração e variação pequena
sessao_ok() {
  local csv="results/experiments/${BASE}/$1/results.csv"
  [[ -f "$csv" ]] || return 1
  awk -F, -v lim="$LIMITE_VARIACAO" '
    NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
    $h["rep_kind"] == "metricas" {
      n++; t = $h["loop_ms"] + 0
      if ($h["run_ok"] != 1 || $h["calib_warn"] > 0 || t <= 0) bad = 1
      if (min == "" || t < min) min = t
      if (t > max) max = t
    }
    END { exit (n == 0 || bad || max / min > lim) ? 1 : 0 }' "$csv"
}

# rodar <sessão> <calibrações> <medições> <args do run-experiments...>
rodar() {
  local sessao="$1" calib="$2" med="$3"; shift 3
  local setup="--skip-setup --no-build --no-sync"
  ((PRIMEIRA)) && setup="" && PRIMEIRA=0
  local cmd="$RUN ${setup} --nodes 1 $CALIB --session ${BASE}/${sessao} --warmup-reps ${calib} --timed-reps ${med} $*"
  log "Sessão ${sessao}"
  if ((DRY_RUN)); then echo "  $cmd"; return; fi
  eval "$cmd" || log "AVISO: run-experiments terminou com erro na sessão ${sessao}"
  if ! sessao_ok "$sessao"; then
    log "Sessão ${sessao} com aviso de calibração, falha ou variação > ${LIMITE_VARIACAO}: repetindo uma vez"
    local dir="results/experiments/${BASE}/${sessao}"
    rm -rf "${dir}_descartada"
    mv "$dir" "${dir}_descartada"
    eval "${cmd/--warmup-reps ${calib}/--warmup-reps 1}" || true
    sessao_ok "$sessao" || log "AVISO: sessão ${sessao} continua com problema; conferir os logs"
  fi
}

inicio=$(date +%s)

# ---- 1. Calibração com todos os pontos ---------------------------------------
# Os 3 cenários usam input.bin (cada geração sobrescreve a anterior, ~10–16 GB cada)
if ((FAZER_CALIBRACAO)); then
  for c in pq deep sift; do
    rodar "calib_${c}" 2 2 --env "cluster-config/cenario_${c}.env" --input input.bin
  done
fi

# ---- 2. Teste de chunks com os pontos de um nó do cluster de 32 -------------
if ((FAZER_CHUNKS)); then
  gerado=""
  for linha in "${CONFIGS[@]}"; do
    read -r c nome chunks peq frac <<< "$linha"
    env_file="cluster-config/cenario_${c}.env"
    n_total="$(grep -m1 '^N=' "$env_file" | cut -d= -f2 | awk '{print $1}')"
    n_no=$((n_total / 32))
    for juntos in 0 1; do
      extra="-x KMEANS_ASSIGN_CALC_JUNTOS=${juntos}"
      ((peq > 0)) && extra+=" -x KMEANS_CHUNKS_PEQUENOS=${peq} -x KMEANS_FRACAO_PEQUENOS=${frac}"
      gerar="--no-generate"
      [[ "$gerado" == *" ${c} "* ]] || { gerar=""; gerado+=" ${c} "; }
      rodar "chunks_${c}_${nome}_j${juntos}" 3 3 --env "$env_file" --n "$n_no" \
        --input "input_${c}_32.bin" $gerar --chunks-total "$chunks" --extra-mpirun-args "'${extra}'"
    done
  done
fi

((DRY_RUN)) && exit 0
log "Concluído em $(( ($(date +%s) - inicio) / 60 )) min"
python3 scripts/resumo_rodada_1no.py "results/experiments/${BASE}" | tee "results/experiments/${BASE}/resumo.txt"
