#!/bin/bash
# =============================================================================
# sanity_check_multinode_omp.sh — Verifica ambiente multi-nodo OpenMP
# Uso: bash scripts/sanity_check_multinode_omp.sh <hostfile> <np>
# Executar a partir da RAIZ do projeto NO NO MESTRE
# =============================================================================

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HOSTFILE="${1:-hostfile}"
NP="${2:-2}"
K=5
ITERS=5
SEED=42
SANITY_INPUT="input_sanity_10k.txt"
LOG_DIR="logs/sanity_multinode_omp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================"
echo " SANITY CHECK MULTI-NODO — OpenMP"
echo " Hostfile : $HOSTFILE"
echo " np       : $NP"
echo " Log dir  : $LOG_DIR"
echo "======================================"

# Verificar hostfile
if [ ! -f "$HOSTFILE" ]; then
    echo "ERRO: hostfile '$HOSTFILE' nao encontrado"
    exit 1
fi

echo ""
echo "Nos no hostfile:"
cat "$HOSTFILE"
echo ""

# Verificar binario
if [ ! -f "./src/openmp/kmeans_openmp" ]; then
    echo "Compilando OpenMP (CPU)..."
    cd src/openmp && make -s && cd "$ROOT_DIR"
fi

# Verificar OMP GPU
HAS_OMP_GPU=0
if command -v nvc++ &> /dev/null; then
    HAS_OMP_GPU=1
    cd src/openmp && make -s GPU=1 && cd "$ROOT_DIR"
else
    echo "AVISO: nvc++ nao encontrado — modos GPU e Hybrid serao ignorados"
fi

# Gerar input pequeno se nao existir
if [ ! -f "$SANITY_INPUT" ]; then
    echo "Gerando dataset pequeno (10k pontos)..."
    python3 generate_input.py 10000 2 $SEED "$SANITY_INPUT" 4
fi

# =============================================================================
# Teste 1: Hostnames distintos
# =============================================================================
echo ""
echo "=== TESTE 1: Verificar hostnames dos nos ==="
mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
    hostname 2>&1 | tee "$LOG_DIR/hostnames.log"

UNIQUE_HOSTS=$(mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none hostname 2>/dev/null | sort -u | wc -l)
if [ "$UNIQUE_HOSTS" -lt "$NP" ]; then
    echo "AVISO: $UNIQUE_HOSTS host(s) unico(s) para $NP rank(s) — possivel problema!"
else
    echo "OK: $UNIQUE_HOSTS hosts distintos — cada rank em uma VM diferente"
fi

# =============================================================================
# Teste 2: OpenMP CPU
# =============================================================================
echo ""
echo "=== TESTE 2: OpenMP CPU (np=$NP, OMP_NUM_THREADS=7) ==="
mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
    -x OMP_NUM_THREADS=7 \
./src/openmp/kmeans_openmp "$SANITY_INPUT" $K "output_sanity/multinode_omp_cpu_np${NP}" 0 1 0.0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/omp_cpu_np${NP}.log"
echo "OpenMP CPU np=$NP: OK"

if [ $HAS_OMP_GPU -eq 1 ]; then
    # =============================================================================
    # Teste 3: OpenMP GPU
    # =============================================================================
    echo ""
    echo "=== TESTE 3: OpenMP GPU (np=$NP, OMP_NUM_THREADS=1) ==="
    mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
        -x OMP_NUM_THREADS=1 \
    ./src/openmp/kmeans_openmp "$SANITY_INPUT" $K "output_sanity/multinode_omp_gpu_np${NP}" 1 2 1.0 $SEED $ITERS \
    2>&1 | tee "$LOG_DIR/omp_gpu_np${NP}.log"
    echo "OpenMP GPU np=$NP: OK"

    # =============================================================================
    # Teste 4: OpenMP Hybrid
    # =============================================================================
    echo ""
    echo "=== TESTE 4: OpenMP Hybrid (np=$NP, OMP_NUM_THREADS=6) ==="
    mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
        -x OMP_NUM_THREADS=6 \
    ./src/openmp/kmeans_openmp "$SANITY_INPUT" $K "output_sanity/multinode_omp_hybrid_np${NP}" 2 4 0.75 $SEED $ITERS \
    2>&1 | tee "$LOG_DIR/omp_hybrid_np${NP}.log"
    echo "OpenMP Hybrid np=$NP: OK"
else
    echo ""
    echo "IGNORADO: OpenMP GPU e Hybrid (nvc++ nao disponivel)"
fi

# =============================================================================
# Resumo
# =============================================================================
echo ""
echo "=== RESUMO ==="
echo "Hosts distintos detectados: $UNIQUE_HOSTS / $NP"
for mode in cpu gpu hybrid; do
    log="$LOG_DIR/omp_${mode}_np${NP}.log"
    [ -f "$log" ] || continue
    SSE=$(grep "SSE" "$log" | awk '{print $NF}')
    RANKS=$(grep "\[RANK" "$log" | sort -u)
    echo "OpenMP ${mode}: SSE=${SSE}"
    echo "$RANKS"
done

echo ""
echo "======================================"
echo " SANITY CHECK MULTI-NODO OMP CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
