#!/bin/bash
# =============================================================================
# run_benchmark_seq.sh — Benchmark versão sequencial
# Uso: bash scripts/run_benchmark_seq.sh [input.txt]
# Executar a partir da RAIZ do projeto
# =============================================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_FILE="${1:-input.txt}"
K=50
SEED=42
ITERS=30
REPS=1
LOG_DIR="logs/benchmark_seq_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================"
echo " BENCHMARK Sequencial — KMeans"
echo " Input : $INPUT_FILE"
echo " Logs  : $LOG_DIR"
echo "======================================"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERRO: '$INPUT_FILE' nao encontrado."
    exit 1
fi

if [ ! -f "./src/sequencial/kmeans_seq" ]; then
    echo "Compilando sequencial..."
    cd src/sequencial && make -s && cd "$ROOT_DIR"
fi

for ((rep=1; rep<=REPS; rep++)); do
    log="$LOG_DIR/seq_metricas_${rep}.log"
    echo "" && echo "[$(date +%H:%M:%S)] >>> Sequencial — Run $rep"
    ./src/sequencial/kmeans_seq "$INPUT_FILE" $K "output_bench/seq_${rep}" $SEED $ITERS \
    2>&1 | tee "$log"
done

echo ""
echo "======================================"
echo " BENCHMARK SEQUENCIAL CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
