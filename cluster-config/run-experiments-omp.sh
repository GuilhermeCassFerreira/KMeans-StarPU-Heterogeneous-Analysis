#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# run-experiments-omp.sh — Benchmark OpenMP multi-nodo
# Uso: bash cluster-config/run-experiments-omp.sh [opcoes]
# Executar a partir da RAIZ do projeto NO NO MESTRE
#
# Pre-requisito: cluster-config/cluster-mpi.sh set-ips "ip1,ip2,..."
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLUSTER_TOOL="${SCRIPT_DIR}/cluster-mpi.sh"

SESSION_NAME="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/experiments_omp/${SESSION_NAME}"

MODES_RAW="omp_cpu omp_gpu omp_hybrid"
NODE_COUNTS_RAW="2"
N=300000000
K=50
SEED=42
ITERS=30
TIMED_REPS=3
DO_SYNC=1
DO_BUILD=1
BUILD_CLEAN=1
GLOBAL_MPIRUN_ARGS="--bind-to none"
SAVE_LOGS=1

# Chunks por nó (aplicados por rank)
CHUNKS_CPU_PER_NODE=1
CHUNKS_GPU_PER_NODE=1
CHUNKS_HYBRID_PER_NODE=4
GPU_RATIO_HYBRID=0.75

POOL_CONTROL_IPS=()
NODE_COUNTS=()
MODES=()

log() { printf '[experiments-omp] %s\n' "$*"; }
die() { printf '[experiments-omp] ERRO: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Uso: $(basename "$0") [opcoes]
  --modes "a b c"     omp_cpu omp_gpu omp_hybrid
  --nodes "2 4"       contagem de nos
  --timed-reps <n>    runs de metricas (default: 3)
  --no-sync           nao sincronizar
  --no-build          nao compilar
  -h, --help
USAGE
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --modes)      MODES_RAW="$2";       shift 2 ;;
      --nodes)      NODE_COUNTS_RAW="$2"; shift 2 ;;
      --timed-reps) TIMED_REPS="$2";      shift 2 ;;
      --no-sync)    DO_SYNC=0;            shift   ;;
      --no-build)   DO_BUILD=0;           shift   ;;
      -h|--help)    usage; exit 0 ;;
      *) die "Argumento desconhecido: $1" ;;
    esac
  done
}

validate_config() {
  [[ -x "$CLUSTER_TOOL" ]] || die "cluster-mpi.sh nao encontrado: $CLUSTER_TOOL"
  read -r -a NODE_COUNTS <<< "$NODE_COUNTS_RAW"
  read -r -a MODES      <<< "$MODES_RAW"

  local control_file="${SCRIPT_DIR}/.cluster/control_ips.txt"
  [[ -f "$control_file" ]] || die "IPs nao configurados. Execute: ./cluster-config/cluster-mpi.sh set-ips ip1,ip2,..."
  mapfile -t POOL_CONTROL_IPS < "$control_file"
}

set_cluster_subset() {
  local nodes="$1"
  local ctl_csv
  ctl_csv="$(IFS=','; echo "${POOL_CONTROL_IPS[*]:0:nodes}")"
  "${CLUSTER_TOOL}" set-ips "$ctl_csv" >/dev/null
}

chunks_for_mode() {
  local mode="$1"
  case "$mode" in
    omp_cpu)    echo "$CHUNKS_CPU_PER_NODE" ;;
    omp_gpu)    echo "$CHUNKS_GPU_PER_NODE" ;;
    omp_hybrid) echo "$CHUNKS_HYBRID_PER_NODE" ;;
  esac
}

extract_loop_time_ms() {
  awk '/Tempo do loop/ {gsub(/[^0-9.]/, "", $NF); print $NF; exit}' <<< "$1"
}

extract_sse() {
  awk '/SSE/ {print $NF; exit}' <<< "$1"
}

run_single() {
  local mode="$1" nodes="$2" chunks="$3" rep_idx="$4"
  local csv_file="$5" logs_dir="$6"

  local omp_mode omp_threads
  case "$mode" in
    omp_cpu)
      omp_mode=0; omp_threads=7 ;;
    omp_gpu)
      omp_mode=1; omp_threads=1 ;;
    omp_hybrid)
      omp_mode=2; omp_threads=6 ;;
  esac

  local gpu_ratio
  [[ "$mode" == "omp_hybrid" ]] && gpu_ratio="$GPU_RATIO_HYBRID" || gpu_ratio="0.0"

  local out_dir="output_bench/exp_${mode}_np${nodes}_metricas${rep_idx}"
  local run_args="input.txt ${K} ${out_dir} ${omp_mode} ${chunks} ${gpu_ratio} ${SEED} ${ITERS}"
  local mpirun_args="${GLOBAL_MPIRUN_ARGS} -x OMP_NUM_THREADS=${omp_threads}"

  log "Run mode=${mode} nodes=${nodes} chunks=${chunks} metricas:${rep_idx} OMP_NUM_THREADS=${omp_threads}"

  local output status
  set +e
  output="$("${CLUSTER_TOOL}" run openmp \
    --mpirun-args "$mpirun_args" \
    --run-args "$run_args" 2>&1)"
  status=$?
  set -e

  printf '%s\n' "$output"

  if [[ "$SAVE_LOGS" -eq 1 ]]; then
    printf '%s\n' "$output" > "${logs_dir}/${mode}_np${nodes}_metricas${rep_idx}.log"
  fi

  local loop_ms sse run_ok
  loop_ms="$(extract_loop_time_ms "$output")"
  sse="$(extract_sse "$output")"
  run_ok=$([[ "$status" -eq 0 && -n "$loop_ms" ]] && echo 1 || echo 0)

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$mode" "$nodes" "$N" "$K" "$chunks" \
    "$rep_idx" "$loop_ms" "$run_ok" \
    >> "$csv_file"
}

main() {
  parse_args "$@"
  validate_config

  local RESULTS_CSV="${RESULTS_DIR}/results.csv"
  local LOGS_DIR="${RESULTS_DIR}/logs"
  mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

  echo "timestamp,mode,nodes,N,K,chunks,rep_idx,loop_ms,run_ok" > "$RESULTS_CSV"

  log "Desabilitando hyperthreading em todos os nos..."
  "${CLUSTER_TOOL}" disable-ht

  log "Gerando dataset no no mestre..."
  "${CLUSTER_TOOL}" generate-input "$N" 2 "$SEED" input.txt 8

  log "Sessao: ${RESULTS_DIR}"

  local nodes mode chunks

  for nodes in "${NODE_COUNTS[@]}"; do
    set_cluster_subset "$nodes"
    [[ "$DO_SYNC" -eq 1 ]] && "${CLUSTER_TOOL}" sync
    if [[ "$DO_BUILD" -eq 1 ]]; then
      local clean_flag=""
      [[ "$BUILD_CLEAN" -eq 1 ]] && clean_flag="--clean"
      "${CLUSTER_TOOL}" build openmp $clean_flag "GPU=1"
    fi

    for mode in "${MODES[@]}"; do
      chunks="$(chunks_for_mode "$mode")"
      for ((rep=1; rep<=TIMED_REPS; rep++)); do
        run_single "$mode" "$nodes" "$chunks" "$rep" "$RESULTS_CSV" "$LOGS_DIR"
      done
    done
  done

  log "Concluido. Resultados em: ${RESULTS_CSV}"
}

main "$@"
