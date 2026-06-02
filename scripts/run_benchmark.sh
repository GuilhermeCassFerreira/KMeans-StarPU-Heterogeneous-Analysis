#!/bin/bash
# =============================================================================
# run_benchmark.sh — Experimentos StarPU N=300M, np=1
# Uso: bash scripts/run_benchmark.sh [input.txt]
# Executar a partir da RAIZ do projeto
#
# StarPU: 5 runs cada modo (1-2 calibracao, 3-5 metricas)
# Ordem intercalada: nunca dois runs do mesmo modo em sequencia
# =============================================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_FILE="${1:-input.txt}"
K=50
SEED=42
ITERS=30
LOG_DIR="logs/benchmark_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"/{starpu_cpu,starpu_gpu,starpu_hybrid}

echo "======================================"
echo " BENCHMARK StarPU — KMeans N=300M, np=1"
echo " Input : $INPUT_FILE"
echo " Logs  : $LOG_DIR"
echo "======================================"

# Verificar input
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERRO: '$INPUT_FILE' nao encontrado."
    echo "Gere com: python3 generate_input.py 300000000 2 42 input.txt 8"
    exit 1
fi

# Compilar se necessario
if [ ! -f "./kmeans_starpu" ]; then
    echo "Compilando StarPU..."
    make -s
fi

# =============================================================================
# Funcoes de execucao
# =============================================================================

run_starpu_cpu() {
    local run_id=$1 label=$2
    local log="$LOG_DIR/starpu_cpu/${run_id}_${label}.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> StarPU CPU — Run $run_id ($label)"
    STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
    STARPU_NCPUS=8 STARPU_NCUDA=0 \
    STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
    mpirun -np 1 --bind-to none \
    ./kmeans_starpu "$INPUT_FILE" $K "output_bench/starpu_cpu_${run_id}" 28 0 $SEED $ITERS \
    2>&1 | tee "$log"
}

run_starpu_gpu() {
    local run_id=$1 label=$2
    local log="$LOG_DIR/starpu_gpu/${run_id}_${label}.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> StarPU GPU — Run $run_id ($label)"
    STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
    STARPU_NCPUS=0 STARPU_NCUDA=1 \
    mpirun -np 1 --bind-to none \
    ./kmeans_starpu "$INPUT_FILE" $K "output_bench/starpu_gpu_${run_id}" 2 0 $SEED $ITERS \
    2>&1 | tee "$log"
}

run_starpu_hybrid() {
    local run_id=$1 label=$2
    local log="$LOG_DIR/starpu_hybrid/${run_id}_${label}.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> StarPU Hybrid — Run $run_id ($label)"
    STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
    STARPU_NCPUS=7 STARPU_NCUDA=1 \
    STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
    mpirun -np 1 --bind-to none \
    ./kmeans_starpu "$INPUT_FILE" $K "output_bench/starpu_hybrid_${run_id}" 32 0 $SEED $ITERS \
    2>&1 | tee "$log"
}

# =============================================================================
# Execucao INTERCALADA — nunca dois runs do mesmo modo em sequencia
#
# FASE 1+2: Calibracao (runs 1 e 2)
# FASE 3+4: Metricas (runs 3, 4 e 5)
# =============================================================================

echo "" && echo "=== FASE 1: Calibracao run 1 ==="
run_starpu_cpu    1 "calibracao"
run_starpu_gpu    1 "calibracao"
run_starpu_hybrid 1 "calibracao"

echo "" && echo "=== FASE 2: Calibracao run 2 ==="
run_starpu_cpu    2 "calibracao"
run_starpu_gpu    2 "calibracao"
run_starpu_hybrid 2 "calibracao"

echo "" && echo "=== FASE 3: Metricas run 1 ==="
run_starpu_cpu    3 "metricas"
run_starpu_gpu    3 "metricas"
run_starpu_hybrid 3 "metricas"

echo "" && echo "=== FASE 4: Metricas runs 2 e 3 ==="
run_starpu_cpu    4 "metricas"
run_starpu_gpu    4 "metricas"
run_starpu_hybrid 4 "metricas"
run_starpu_cpu    5 "metricas"
run_starpu_gpu    5 "metricas"
run_starpu_hybrid 5 "metricas"

echo ""
echo "======================================"
echo " BENCHMARK CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
