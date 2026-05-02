#!/bin/bash

# --- CONFIGURAÇÕES DE AMBIENTE ---
export PMIX_MCA_gds=hash
export STARPU_WORKERS_GETBIND=0
# Adicionamos todos os caminhos de bibliotecas necessários
export LD_LIBRARY_PATH=/home/bridge/starpu_install/lib:/usr/local/cuda/lib64:/opt/openmpi/lib:$LD_LIBRARY_PATH

# --- PARÂMETROS DO SISTEMA ---
# Você pode alterar esses valores aqui ou passar via linha de comando
INPUT_FILE=${1:-"input.txt"}
K_CLUSTERS=${2:-1000}
OUTPUT_DIR=${3:-"o"}
CHUNK_SIZE=$4  # Opcional

# --- COMANDO DE EXECUÇÃO ---
echo "🚀 Iniciando Cluster StarPU-MPI..."
echo "📍 Arquivo: $INPUT_FILE | K: $K_CLUSTERS"

/opt/openmpi/bin/mpirun --prefix /opt/openmpi \
    --bind-to none \
    --mca pml ob1 \
    --mca btl tcp,self \
    --mca btl_tcp_if_include 192.168.0.0/24 \
    -x STARPU_WORKERS_GETBIND=0 \
    -x LD_LIBRARY_PATH \
    -x PMIX_MCA_gds=hash \
    -hostfile hostfile \
    -np 2 \
    ./kmeans_starpu $INPUT_FILE $K_CLUSTERS $OUTPUT_DIR $CHUNK_SIZE