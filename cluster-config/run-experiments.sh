#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# run-experiments.sh — Benchmark KMeans StarPU multi-nodo
# Uso: bash cluster-config/run-experiments.sh [opcoes]
# Executar a partir da RAIZ do projeto
#
# Pre-requisito: cluster-config/cluster-mpi.sh set-ips "ip1,ip2,..."
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLUSTER_TOOL="${SCRIPT_DIR}/cluster-mpi.sh"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/experiments.env"

ENV_FILE="${DEFAULT_ENV_FILE}"
SESSION_NAME="$(date +%Y%m%d-%H%M%S)"
RESULTS_BASE_DIR="${REPO_ROOT}/results/experiments"
RESULTS_DIR=""

MODES_RAW="starpu_cpu starpu_gpu starpu_hybrid"
NODE_COUNTS_RAW="2 4"
N=300000000
K=50
SEED=42
ITERS=30
CHUNKS_CPU_PER_NODE=28
CHUNKS_GPU_PER_NODE=2
CHUNKS_HYBRID_PER_NODE=32
WARMUP_REPS=2
TIMED_REPS=3
DO_SYNC=1
DO_BUILD=1
BUILD_CLEAN=1
BUILD_ARGS_STARPU=""
GLOBAL_MPIRUN_ARGS="--bind-to none"
SAVE_LOGS=1

POOL_CONTROL_IPS=()
POOL_MPI_IPS=()
NODE_COUNTS=()
MODES=()

usage() {
  cat <<USAGE
Uso: $(basename "$0") [opcoes]

Opcoes:
  --env <file>           Arquivo de configuracao (default: cluster-config/experiments.env)
  --session <nome>       Nome da sessao de resultados (default: timestamp)
  --modes "a b c"        Modos: starpu_cpu starpu_gpu starpu_hybrid
  --nodes "2 4"          Contagens de nos MPI
  --warmup-reps <n>      Runs de calibracao por cenario (default: 2)
  --timed-reps <n>       Runs de metricas por cenario (default: 3)
  --no-sync              Nao sincronizar repositorio
  --no-build             Nao compilar
  -h, --help             Mostrar ajuda
USAGE
}

log() { printf '[experiments] %s\n' "$*"; }
die() { printf '[experiments] ERRO: %s\n' "$*" >&2; exit 1; }

load_env_file() {
  [[ -f "$ENV_FILE" ]] && source "$ENV_FILE" || true
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --env)        ENV_FILE="$2";       shift 2 ;;
      --session)    SESSION_NAME="$2";   shift 2 ;;
      --modes)      MODES_RAW="$2";      shift 2 ;;
      --nodes)      NODE_COUNTS_RAW="$2"; shift 2 ;;
      --warmup-reps) WARMUP_REPS="$2";  shift 2 ;;
      --timed-reps)  TIMED_REPS="$2";   shift 2 ;;
      --no-sync)    DO_SYNC=0;           shift   ;;
      --no-build)   DO_BUILD=0;          shift   ;;
      -h|--help)    usage; exit 0 ;;
      *) die "Argumento desconhecido: $1" ;;
    esac
  done
}

validate_config() {
  [[ -x "$CLUSTER_TOOL" ]] || die "cluster-mpi.sh nao encontrado ou nao executavel: $CLUSTER_TOOL"

  read -r -a NODE_COUNTS <<< "$NODE_COUNTS_RAW"
  read -r -a MODES      <<< "$MODES_RAW"

  ((${#NODE_COUNTS[@]} > 0)) || die "NODE_COUNTS vazio"
  ((${#MODES[@]} > 0))       || die "MODES vazio"

  local control_file="${SCRIPT_DIR}/.cluster/control_ips.txt"
  [[ -f "$control_file" ]] || die "IPs nao configurados. Execute: ./cluster-config/cluster-mpi.sh set-ips ip1,ip2,..."

  mapfile -t POOL_CONTROL_IPS < "$control_file"
  local mpi_file="${SCRIPT_DIR}/.cluster/mpi_ips.txt"
  if [[ -f "$mpi_file" && -s "$mpi_file" ]]; then
    mapfile -t POOL_MPI_IPS < "$mpi_file"
  else
    POOL_MPI_IPS=("${POOL_CONTROL_IPS[@]}")
  fi

  ((${#POOL_CONTROL_IPS[@]} > 0)) || die "Nenhum IP de controle encontrado"

  local max_nodes=0
  for n in "${NODE_COUNTS[@]}"; do
    ((n > max_nodes)) && max_nodes=$n
  done
  ((${#POOL_CONTROL_IPS[@]} >= max_nodes)) || \
    die "Precisa de ${max_nodes} nos; apenas ${#POOL_CONTROL_IPS[@]} disponiveis"

  [[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="${RESULTS_BASE_DIR}/${SESSION_NAME}"
}

set_cluster_subset() {
  local nodes="$1"
  local ctl_csv mpi_csv
  ctl_csv="$(IFS=','; echo "${POOL_CONTROL_IPS[*]:0:nodes}")"
  mpi_csv="$(IFS=','; echo "${POOL_MPI_IPS[*]:0:nodes}")"
  log "Selecionando ${nodes} nos"
  "${CLUSTER_TOOL}" set-ips "$ctl_csv" "$mpi_csv" >/dev/null
}

chunks_for_mode() {
  local mode="$1" nodes="$2"
  case "$mode" in
    starpu_cpu)    echo $((CHUNKS_CPU_PER_NODE    * nodes)) ;;
    starpu_gpu)    echo $((CHUNKS_GPU_PER_NODE    * nodes)) ;;
    starpu_hybrid) echo $((CHUNKS_HYBRID_PER_NODE * nodes)) ;;
  esac
}

starpu_env_for_mode() {
  local mode="$1"
  case "$mode" in
    starpu_cpu)
      echo "STARPU_NCPUS=8 STARPU_NCUDA=0 STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7"
      ;;
    starpu_gpu)
      echo "STARPU_NCPUS=0 STARPU_NCUDA=1"
      ;;
    starpu_hybrid)
      echo "STARPU_NCPUS=7 STARPU_NCUDA=1 STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7"
      ;;
  esac
}

build_for_nodes() {
  local nodes="$1"
  if [[ "$DO_BUILD" -eq 0 ]]; then
    log "Pulando build"
    return
  fi

  local clean_flag=""
  [[ "$BUILD_CLEAN" -eq 1 ]] && clean_flag="--clean"

  "${CLUSTER_TOOL}" build starpu $clean_flag $BUILD_ARGS_STARPU
}

extract_loop_time_ms() {
  local output="$1"
  awk '/Tempo do loop/ {gsub(/[^0-9.]/, "", $NF); print $NF; exit}' <<< "$output"
}

extract_sse() {
  local output="$1"
  awk '/SSE/ {print $NF; exit}' <<< "$output"
}

run_single() {
  local mode="$1" nodes="$2" chunks="$3" rep_kind="$4" rep_idx="$5"
  local csv_file="$6" logs_dir="$7"

  local starpu_env
  starpu_env="$(starpu_env_for_mode "$mode")"

  local out_dir="output_bench/exp_${mode}_np${nodes}_${rep_kind}${rep_idx}"
  local run_args="input.txt ${K} ${out_dir} ${chunks} 0 ${SEED} ${ITERS}"

  local mpirun_args="${GLOBAL_MPIRUN_ARGS} -x STARPU_SCHED=dmda -x STARPU_NOPENCL=1 -x STARPU_RESERVE_NCPU=1"
  for var in $starpu_env; do
    mpirun_args+=" -x ${var}"
  done

  log "Run mode=${mode} nodes=${nodes} chunks=${chunks} ${rep_kind}:${rep_idx}"

  local output status
  set +e
  output="$("${CLUSTER_TOOL}" run starpu \
    --mpirun-args "$mpirun_args" \
    --run-args "$run_args" 2>&1)"
  status=$?
  set -e

  printf '%s\n' "$output"

  if [[ "$SAVE_LOGS" -eq 1 ]]; then
    local log_file="${logs_dir}/${mode}_np${nodes}_${rep_kind}${rep_idx}.log"
    printf '%s\n' "$output" > "$log_file"
  fi

  local loop_ms sse run_ok notes
  loop_ms="$(extract_loop_time_ms "$output")"
  sse="$(extract_sse "$output")"
  run_ok="1"
  notes=""

  if [[ "$status" -ne 0 || -z "$loop_ms" ]]; then
    run_ok="0"
    notes="exit_status_${status}"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$mode" "$nodes" "$N" "$K" "$chunks" \
    "$rep_kind" "$rep_idx" \
    "$loop_ms" "$sse" "$run_ok" \
    >> "$csv_file"
}

main() {
  load_env_file
  parse_args "$@"
  validate_config

  local RESULTS_CSV="${RESULTS_DIR}/results.csv"
  local LOGS_DIR="${RESULTS_DIR}/logs"
  mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

  echo "timestamp,mode,nodes,N,K,chunks,rep_kind,rep_idx,loop_ms,sse,run_ok" > "$RESULTS_CSV"

  # Desabilitar HT em todos os nos
  log "Desabilitando hyperthreading em todos os nos..."
  "${CLUSTER_TOOL}" disable-ht

  # Gerar input.txt apenas no no mestre (rank 0)
  log "Gerando dataset no no mestre..."
  "${CLUSTER_TOOL}" generate-input "$N" 2 "$SEED" input.txt 8

  log "Sessao: ${RESULTS_DIR}"

  local nodes mode chunks total_reps rep rep_kind rep_idx

  for nodes in "${NODE_COUNTS[@]}"; do
    set_cluster_subset "$nodes"
    [[ "$DO_SYNC" -eq 1 ]] && "${CLUSTER_TOOL}" sync
    build_for_nodes "$nodes"

    total_reps=$((WARMUP_REPS + TIMED_REPS))
    for mode in "${MODES[@]}"; do
      chunks="$(chunks_for_mode "$mode" "$nodes")"
      for ((rep=1; rep<=total_reps; rep++)); do
        if ((rep <= WARMUP_REPS)); then
          rep_kind="calibracao"; rep_idx="$rep"
        else
          rep_kind="metricas"; rep_idx="$((rep - WARMUP_REPS))"
        fi
        run_single "$mode" "$nodes" "$chunks" "$rep_kind" "$rep_idx" "$RESULTS_CSV" "$LOGS_DIR"
      done
    done
  done

  log "Concluido. Resultados em: ${RESULTS_CSV}"
}

main "$@"
