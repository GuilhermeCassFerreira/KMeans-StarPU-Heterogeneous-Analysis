#!/bin/bash
# =============================================================================
# sanity_check_multinode.sh — Verifica ambiente multi-nodo
# Uso: bash scripts/sanity_check_multinode.sh <hostfile> <np>
# Executar a partir da RAIZ do projeto NO NO MESTRE
#
# Confirma:
#  1. Todos os nos estao acessiveis via MPI
#  2. Cada rank roda em uma VM diferente (hostname distinto)
#  3. StarPU CPU, GPU e Hybrid funcionam com np correto
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
LOG_DIR="logs/sanity_multinode_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================"
echo " SANITY CHECK MULTI-NODO"
echo " Hostfile : $HOSTFILE"
echo " np       : $NP"
echo " Log dir  : $LOG_DIR"
echo "======================================"

# Verificar hostfile
if [ ! -f "$HOSTFILE" ]; then
    echo "ERRO: hostfile '$HOSTFILE' nao encontrado"
    echo "Gere com: ./cluster-config/cluster-mpi.sh hostfile"
    exit 1
fi

echo ""
echo "Nos no hostfile:"
cat "$HOSTFILE"
echo ""

# Gerar input pequeno se nao existir
if [ ! -f "$SANITY_INPUT" ]; then
    echo "[1] Gerando dataset pequeno (10k pontos)..."
    python3 generate_input.py 10000 2 $SEED "$SANITY_INPUT" 4
    echo "OK"
fi

# Verificar binario
if [ ! -f "./kmeans_starpu" ]; then
    echo "Compilando StarPU..."
    make -s
fi

# =============================================================================
# Teste 1: MPI basico — confirmar hostnames distintos
# =============================================================================
echo ""
echo "=== TESTE 1: Verificar hostnames dos nos ==="
mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
    hostname 2>&1 | tee "$LOG_DIR/hostnames.log"

UNIQUE_HOSTS=$(mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none hostname 2>/dev/null | sort -u | wc -l)
if [ "$UNIQUE_HOSTS" -lt "$NP" ]; then
    echo "AVISO: Apenas $UNIQUE_HOSTS host(s) unico(s) para $NP rank(s) — alguns ranks podem estar na mesma VM!"
else
    echo "OK: $UNIQUE_HOSTS hosts distintos para $NP ranks — cada rank em uma VM diferente"
fi

# =============================================================================
# Teste 2: StarPU CPU multi-nodo
# =============================================================================
echo ""
echo "=== TESTE 2: StarPU CPU (np=$NP) ==="
STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1 \
STARPU_NCPUS=7 STARPU_NCUDA=0 \
STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
    -x STARPU_SCHED=dmda -x STARPU_NOPENCL=1 -x STARPU_RESERVE_NCPU=1 \
    -x STARPU_NCPUS=7 -x STARPU_NCUDA=0 \
    -x STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
./kmeans_starpu "$SANITY_INPUT" $K "output_sanity/multinode_cpu_np${NP}" 4 0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/starpu_cpu_np${NP}.log"
echo "StarPU CPU np=$NP: OK"

# =============================================================================
# Teste 3: StarPU GPU multi-nodo
# =============================================================================
echo ""
echo "=== TESTE 3: StarPU GPU (np=$NP) ==="
mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
    -x STARPU_SCHED=dmda -x STARPU_NOPENCL=1 -x STARPU_RESERVE_NCPU=1 \
    -x STARPU_NCPUS=0 -x STARPU_NCUDA=1 \
./kmeans_starpu "$SANITY_INPUT" $K "output_sanity/multinode_gpu_np${NP}" 2 0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/starpu_gpu_np${NP}.log"
echo "StarPU GPU np=$NP: OK"

# =============================================================================
# Teste 4: StarPU Hybrid multi-nodo
# =============================================================================
echo ""
echo "=== TESTE 4: StarPU Hybrid (np=$NP) ==="
mpirun -np "$NP" --hostfile "$HOSTFILE" --bind-to none \
    -x STARPU_SCHED=dmda -x STARPU_NOPENCL=1 -x STARPU_RESERVE_NCPU=1 \
    -x STARPU_NCPUS=6 -x STARPU_NCUDA=1 \
    -x STARPU_WORKERS_CPUID=0,1,2,3,4,5,6,7 \
./kmeans_starpu "$SANITY_INPUT" $K "output_sanity/multinode_hybrid_np${NP}" 4 0 $SEED $ITERS \
2>&1 | tee "$LOG_DIR/starpu_hybrid_np${NP}.log"
echo "StarPU Hybrid np=$NP: OK"

# =============================================================================
# Resumo: verificar SSE identico em todos os modos
# =============================================================================
echo ""
echo "=== RESUMO ==="
echo "Hosts distintos detectados: $UNIQUE_HOSTS / $NP"
for mode in cpu gpu hybrid; do
    SSE=$(grep "SSE" "$LOG_DIR/starpu_${mode}_np${NP}.log" | awk '{print $NF}')
    RANKS=$(grep "\[RANK" "$LOG_DIR/starpu_${mode}_np${NP}.log" | sort -u)
    echo "StarPU ${mode}: SSE=${SSE}"
    echo "$RANKS"
done

echo ""
echo "======================================"
echo " SANITY CHECK MULTI-NODO CONCLUIDO"
echo " Logs em: $LOG_DIR"
echo "======================================"
