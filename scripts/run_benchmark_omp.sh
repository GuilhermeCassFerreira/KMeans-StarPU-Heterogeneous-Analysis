#!/bin/bash
# =============================================================================
# run_benchmark_omp.sh — Experimentos OpenMP N=300M, np=1
# Uso: bash scripts/run_benchmark_omp.sh [input.txt]
# Executar a partir da RAIZ do projeto
#
# OpenMP: 3 runs cada modo (todas metricas)
# Ordem intercalada: nunca dois runs do mesmo modo em sequencia
# =============================================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_FILE="${1:-input.txt}"
K=50
SEED=42
ITERS=30
LOG_DIR="logs/benchmark_omp_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"/{omp_cpu,omp_hybrid}

echo "======================================"
echo " BENCHMARK OpenMP — KMeans N=300M, np=1"
echo " Input : $INPUT_FILE"
echo " Logs  : $LOG_DIR"
echo "======================================"

# Verificar input
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERRO: '$INPUT_FILE' nao encontrado."
    echo "Gere com: python3 generate_input.py 300000000 2 42 input.txt 8"
    exit 1
fi

# Compilar OpenMP CPU se necessario
if [ ! -f "./src/openmp/kmeans_openmp" ]; then
    echo "Compilando OpenMP (CPU)..."
    cd src/openmp && make -s && cd "$ROOT_DIR"
fi

# Verificar OMP GPU
HAS_OMP_GPU=0
if command -v nvc++ &> /dev/null; then
    HAS_OMP_GPU=1
    echo "Compilando OpenMP (GPU)..."
    cd src/openmp && make -s GPU=1 && cd "$ROOT_DIR"
else
    echo "AVISO: nvc++ nao encontrado — runs OpenMP GPU/Hybrid serao ignorados"
fi

# =============================================================================
# Funcoes de execucao
# =============================================================================

run_omp_cpu() {
    local run_id=$1
    local log="$LOG_DIR/omp_cpu/${run_id}_metricas.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> OpenMP CPU — Run $run_id"
    OMP_NUM_THREADS=7 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp "$INPUT_FILE" $K "output_bench/omp_cpu_${run_id}" 0 1 0.0 $SEED $ITERS \
    2>&1 | tee "$log"
}

run_omp_gpu() {
    local run_id=$1
    local log="$LOG_DIR/omp_gpu/${run_id}_metricas.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> OpenMP GPU — Run $run_id"
    if [ $HAS_OMP_GPU -eq 0 ]; then echo "IGNORADO: nvc++ nao disponivel"; return; fi
    OMP_NUM_THREADS=1 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp "$INPUT_FILE" $K "output_bench/omp_gpu_${run_id}" 1 2 1.0 $SEED $ITERS \
    2>&1 | tee "$log"
}

run_omp_hybrid() {
    local run_id=$1
    local log="$LOG_DIR/omp_hybrid/${run_id}_metricas.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> OpenMP Hybrid — Run $run_id"
    if [ $HAS_OMP_GPU -eq 0 ]; then echo "IGNORADO: nvc++ nao disponivel"; return; fi
    OMP_NUM_THREADS=6 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp "$INPUT_FILE" $K "output_bench/omp_hybrid_${run_id}" 2 4 0.75 $SEED $ITERS \
    2>&1 | tee "$log"
}

# =============================================================================
# Execucao INTERCALADA — nunca dois runs do mesmo modo em sequencia
# 3 runs x 3 modos = 9 runs total
# =============================================================================

echo "" && echo "=== RODADA 1 ==="
run_omp_cpu    1
run_omp_hybrid 1

echo "" && echo "=== RODADA 2 ==="
run_omp_cpu    2
run_omp_hybrid 2

echo "" && echo "=== RODADA 3 ==="
run_omp_cpu    3
run_omp_hybrid 3

echo ""
echo "======================================"
echo " BENCHMARK OMP CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
