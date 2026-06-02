#!/bin/bash
# =============================================================================
# sanity_check_omp.sh — Verifica se OpenMP roda corretamente (dataset minusculo)
# Uso: bash scripts/sanity_check_omp.sh
# Executar a partir da RAIZ do projeto
# =============================================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SANITY_INPUT="input_sanity_10k.txt"
K=5
ITERS=5
SEED=42
LOG_DIR="logs/sanity_omp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================"
echo " SANITY CHECK — OpenMP"
echo " Log dir: $LOG_DIR"
echo "======================================"

# 1. Gerar input pequeno
echo ""
echo "[1] Gerando dataset pequeno (10k pontos)..."
python3 generate_input.py 10000 2 $SEED "$SANITY_INPUT" 4
echo "OK: $SANITY_INPUT criado"

# 2. Compilar OpenMP CPU
echo ""
echo "[2] Compilando OpenMP (CPU)..."
cd src/openmp
make -s 2>&1 | tee "$ROOT_DIR/$LOG_DIR/compile_omp_cpu.log"
cd "$ROOT_DIR"
if [ ! -f "./src/openmp/kmeans_openmp" ]; then
    echo "ERRO: kmeans_openmp nao encontrado apos compilacao"
    exit 1
fi
echo "OK: kmeans_openmp (CPU) compilado"

# 3. Verificar OMP GPU (requer nvc++)
HAS_OMP_GPU=0
echo ""
echo "[3] Verificando suporte OpenMP GPU (nvc++)..."
if command -v nvc++ &> /dev/null; then
    echo "nvc++ encontrado — compilando com GPU=1..."
    cd src/openmp
    make -s GPU=1 2>&1 | tee "$ROOT_DIR/$LOG_DIR/compile_omp_gpu.log"
    cd "$ROOT_DIR"
    HAS_OMP_GPU=1
    echo "OK: kmeans_openmp (GPU) compilado"
else
    echo "AVISO: nvc++ nao encontrado — OpenMP GPU/Hybrid sera ignorado"
fi

# =============================================================================
# Testes OpenMP
# =============================================================================

echo ""
echo "--- OpenMP CPU (modo=0, chunks=1) ---"
OMP_NUM_THREADS=8 mpirun -np 1 --bind-to none \
./src/openmp/kmeans_openmp "$SANITY_INPUT" $K "output_sanity/omp_cpu" 0 1 0.0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/sanity_omp_cpu.log"
echo "OpenMP CPU: OK"

if [ $HAS_OMP_GPU -eq 1 ]; then
    echo ""
    echo "--- OpenMP GPU (modo=1, chunks=2) ---"
    OMP_NUM_THREADS=8 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp "$SANITY_INPUT" $K "output_sanity/omp_gpu" 1 2 1.0 $SEED $ITERS \
    2>&1 | tee "$LOG_DIR/sanity_omp_gpu.log"
    echo "OpenMP GPU: OK"

    echo ""
    echo "--- OpenMP Hybrid (modo=2, chunks=4, gpu_ratio=0.75) ---"
    OMP_NUM_THREADS=8 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp "$SANITY_INPUT" $K "output_sanity/omp_hybrid" 2 4 0.75 $SEED $ITERS \
    2>&1 | tee "$LOG_DIR/sanity_omp_hybrid.log"
    echo "OpenMP Hybrid: OK"
else
    echo ""
    echo "IGNORADO: OpenMP GPU e Hybrid (nvc++ nao disponivel)"
fi

echo ""
echo "======================================"
echo " SANITY CHECK OMP CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
