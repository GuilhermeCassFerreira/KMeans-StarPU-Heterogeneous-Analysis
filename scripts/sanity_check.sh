#!/bin/bash
# =============================================================================
# sanity_check.sh — Verifica se StarPU roda corretamente (dataset minusculo)
# Uso: bash scripts/sanity_check.sh
# Executar a partir da RAIZ do projeto
# =============================================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SANITY_INPUT="input_sanity_10k.txt"
K=5
ITERS=5
SEED=42
LOG_DIR="logs/sanity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================"
echo " SANITY CHECK — StarPU"
echo " Log dir: $LOG_DIR"
echo "======================================"

# 1. Gerar input pequeno
echo ""
echo "[1] Gerando dataset pequeno (10k pontos)..."
python3 generate_input.py 10000 2 $SEED "$SANITY_INPUT" 4
echo "OK: $SANITY_INPUT criado"

# 2. Compilar StarPU
echo ""
echo "[2] Compilando StarPU..."
make -s 2>&1 | tee "$LOG_DIR/compile_starpu.log"
if [ ! -f "./kmeans_starpu" ]; then
    echo "ERRO: kmeans_starpu nao encontrado apos compilacao"
    exit 1
fi
echo "OK: kmeans_starpu compilado"

# =============================================================================
# Testes StarPU
# =============================================================================

echo ""
echo "--- StarPU CPU ---"
STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
STARPU_NCPUS=7 STARPU_NCUDA=0 \
STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
mpirun -np 1 --bind-to none \
./kmeans_starpu "$SANITY_INPUT" $K "output_sanity/starpu_cpu" 4 0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/sanity_starpu_cpu.log"
echo "StarPU CPU: OK"

echo ""
echo "--- StarPU GPU ---"
STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
STARPU_NCPUS=0 STARPU_NCUDA=1 \
mpirun -np 1 --bind-to none \
./kmeans_starpu "$SANITY_INPUT" $K "output_sanity/starpu_gpu" 2 0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/sanity_starpu_gpu.log"
echo "StarPU GPU: OK"

echo ""
echo "--- StarPU Hybrid ---"
STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
STARPU_NCPUS=6 STARPU_NCUDA=1 \
STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
mpirun -np 1 --bind-to none \
./kmeans_starpu "$SANITY_INPUT" $K "output_sanity/starpu_hybrid" 4 0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/sanity_starpu_hybrid.log"
echo "StarPU Hybrid: OK"

echo ""
echo "======================================"
echo " SANITY CHECK CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
