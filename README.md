# KMeans Heterogêneo — StarPU, OpenMP e Sequencial

Implementação e análise comparativa do algoritmo K-Means em três versões: **sequencial**, **paralela com OpenMP+MPI** e **heterogênea com StarPU+MPI+CUDA**. Desenvolvido como trabalho de mestrado para avaliar o desempenho em ambientes com múltiplas CPUs e GPUs.

---

## Sumário

- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação de Dependências](#instalação-de-dependências)
- [Compilação](#compilação)
- [Geração do Dataset](#geração-do-dataset)
- [Uso dos Binários](#uso-dos-binários)
- [Variáveis de Ambiente](#variáveis-de-ambiente)
- [Scripts de Benchmark](#scripts-de-benchmark)
- [Formato dos Arquivos](#formato-dos-arquivos)
- [Métricas Coletadas](#métricas-coletadas)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Configuração de Hardware Alvo](#configuração-de-hardware-alvo)
- [Troubleshooting](#troubleshooting)

---

## Arquitetura do Projeto

O projeto implementa K-Means em três paradigmas de paralelismo:

```
                        ARQUIVO DE ENTRADA
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
    Sequencial           OpenMP + MPI          StarPU + MPI
    (1 thread,          (múltiplas CPUs        (task-based,
    sem MPI)            + GPU via               DMDA scheduler,
                        OpenACC/CUDA)           CPU + GPU)
          │                    │                    │
     kmeans_seq          kmeans_openmp        kmeans_starpu
          └────────────────────┴────────────────────┘
                               │
                    ARQUIVOS DE SAÍDA (formato idêntico)
                    K-points.txt  /  K-clusters.txt
```

### Algoritmo K-Means (3 fases por iteração)

| Fase | Operação | Critério de parada |
|------|----------|--------------------|
| 1 — Atribuição | Cada ponto → centroide mais próximo (distância Euclidiana²) | `changes == 0` → convergência |
| 2 — Acumulação | Soma parcial das coordenadas por cluster | — |
| 3 — Atualização | `centroide = soma / contagem` | — |

### Modos de execução — OpenMP

| Modo | Valor | Descrição |
|------|-------|-----------|
| CPU | `0` | Todos os chunks processados nas CPUs (OpenMP) |
| GPU | `1` | Todos os chunks processados na GPU (CUDA/OpenACC) |
| Híbrido | `2` | `gpu_chunks = int(num_chunks × gpu_ratio)` primeiros chunks na GPU, restante na CPU — **execução sequencial** |

> **Atenção Híbrido:** com 2 chunks e `gpu_ratio=0.70`, `int(2×0.70) = 1` → 50% GPU + 50% CPU. Para obter 75% GPU são necessários ≥ 4 chunks.

### Modos de execução — StarPU

| Configuração | `STARPU_NCPUS` | `STARPU_NCUDA` | Chunks recomendados |
|---|---|---|---|
| CPU-only | `N` | `0` | `28` |
| GPU-only | `0` | `1` | `1` (mantém dados na GPU entre iterações) |
| Híbrido | `N` | `1` | `32` (DMDA balanceia dinamicamente) |

---

## Pré-requisitos

| Componente | Versão mínima | Obrigatório |
|------------|---------------|-------------|
| GCC / G++ | 7+ | Sim |
| OpenMPI | 4.0+ | Sim |
| StarPU | 1.4 | Sim (versão StarPU) |
| CUDA Toolkit | 11+ | Não (GPU opcional) |
| NVIDIA HPC SDK | 22+ | Não (OpenMP GPU) |
| Python 3 | 3.6+ | Geração de datasets |

### Verificações rápidas

```bash
# MPI
mpicxx --version

# CUDA
nvcc --version && nvidia-smi

# StarPU
starpu_machine_display | grep -E "CPU|CUDA"
```

---

## Instalação de Dependências

### Sistema (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential git \
    libopenmpi-dev openmpi-bin \
    ocl-icd-opencl-dev libhwloc-dev \
    pkg-config wget
```

### StarPU 1.4

```bash
wget https://files.inria.fr/starpu/starpu-1.4.0/starpu-1.4.0.tar.gz
tar xzf starpu-1.4.0.tar.gz
cd starpu-1.4.0

./configure \
    --prefix=/home/bridge/starpu_install \
    --enable-cuda \
    --enable-mpi \
    --disable-opencl

make -j$(nproc)
make install
cd ..
```

### Variáveis de ambiente permanentes (~/.bashrc)

```bash
export STARPU_LIB="/home/bridge/starpu_install/lib"
export CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${CUDA_LIB}:${STARPU_LIB}:${LD_LIBRARY_PATH}"
```

---

## Compilação

### Sequencial

```bash
cd src/sequencial
make clean && make
# Produz: ./kmeans_seq
```

### OpenMP + MPI

```bash
cd src/openmp

# Sem GPU (apenas CPU com OpenMP)
make clean && make

# Com GPU (requer NVIDIA HPC SDK em /opt/nvidia/hpc_sdk/...)
make clean && make GPU=1
# Produz: ./kmeans_openmp
```

> O Makefile detecta automaticamente o compilador `nvc++` do HPC SDK. Sem ele, usa `mpicxx` + `-fopenmp` (modo CPU somente).

### StarPU + MPI + CUDA

```bash
# Na raiz do projeto
make clean && make
# Produz: ./kmeans_starpu
```

> O Makefile raiz detecta automaticamente o `nvcc`. Se presente, compila com `USE_CUDA=1` e inclui `kmeans_cuda.cu`. Sem CUDA, compila apenas para CPU.

### Compilação completa (tudo de uma vez)

```bash
make clean && make                          # StarPU
make -C src/sequencial clean && make -C src/sequencial
make -C src/openmp clean && make -C src/openmp GPU=1
```

---

## Geração do Dataset

O script `generate_input.py` gera datasets paralelos usando `multiprocessing`:

```bash
# Edite o script e ajuste os parâmetros, depois execute:
python3 generate_input.py
```

Ou gere diretamente via linha de comando:

```bash
python3 -c "
import random
N, D = 100_000_000, 2
with open('input_100M_2d.txt', 'w') as f:
    f.write(f'{N} {D}\n')
    for _ in range(N):
        f.write(' '.join(f'{random.uniform(-1000,1000):.6f}' for _ in range(D)) + '\n')
"
```

### Datasets de referência usados nos experimentos

| Arquivo | Pontos | Dimensões | Tamanho aprox. |
|---------|--------|-----------|----------------|
| `input_1M_2d.txt` | 1 M | 2 | ~34 MB |
| `input_10M_2d.txt` | 10 M | 2 | ~340 MB |
| `input_100M_2d.txt` | 100 M | 2 | ~3,4 GB |
| `input_200M_2d.txt` | 200 M | 2 | ~6,8 GB |
| `input_300M_2d.txt` | 300 M | 2 | ~10,2 GB |

---

## Uso dos Binários

### Sequencial — `kmeans_seq`

```bash
./src/sequencial/kmeans_seq <INPUT> <K> <OUT-DIR> [SEED] [NITERS]
```

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `INPUT` | string | — | Caminho para o arquivo de entrada |
| `K` | int | — | Número de clusters |
| `OUT-DIR` | string | — | Diretório/prefixo de saída |
| `SEED` | int | `42` | Semente aleatória para inicialização dos centroides |
| `NITERS` | int | `100` | Número máximo de iterações |

```bash
# Exemplo
./src/sequencial/kmeans_seq input_100M_2d.txt 50 output/seq 42 30
```

---

### OpenMP + MPI — `kmeans_openmp`

```bash
mpirun -np <NP> ./src/openmp/kmeans_openmp \
    <INPUT> <K> <OUT-DIR> [MODE] [CHUNKS] [GPU_RATIO] [SEED] [NITERS]
```

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `INPUT` | string | — | Caminho para o arquivo de entrada |
| `K` | int | — | Número de clusters |
| `OUT-DIR` | string | — | Diretório/prefixo de saída |
| `MODE` | int | `0` | `0`=CPU, `1`=GPU, `2`=Híbrido |
| `CHUNKS` | int | auto | Número de chunks por rank |
| `GPU_RATIO` | float | `0.5` | Fração dos chunks alocados na GPU (modo 2) |
| `SEED` | int | `42` | Semente aleatória |
| `NITERS` | int | `100` | Máximo de iterações |

```bash
# CPU puro (1 nó, 7 threads)
OMP_NUM_THREADS=7 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp input_100M_2d.txt 50 output/omp_cpu 0 1 0.5 42 30

# GPU puro
OMP_NUM_THREADS=7 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp input_100M_2d.txt 50 output/omp_gpu 1 1 0.5 42 30

# Híbrido 70% GPU (4 chunks: 3 GPU + 1 CPU)
OMP_NUM_THREADS=7 mpirun -np 1 --bind-to none \
    ./src/openmp/kmeans_openmp input_100M_2d.txt 50 output/omp_hybrid 2 4 0.70 42 30
```

---

### StarPU + MPI + CUDA — `kmeans_starpu`

```bash
mpirun -np <NP> ./kmeans_starpu \
    <INPUT> <K> <OUT-DIR> [NUM_CHUNKS] [DYNAMIC_SCHED] [SEED] [ITERS]
```

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `INPUT` | string | — | Caminho para o arquivo de entrada |
| `K` | int | — | Número de clusters |
| `OUT-DIR` | string | — | Diretório/prefixo de saída |
| `NUM_CHUNKS` | int | auto | Número de chunks de tarefas |
| `DYNAMIC_SCHED` | int | `0` | `0`=estático, `1`=dinâmico |
| `SEED` | int | `42` | Semente aleatória |
| `ITERS` | int | `100` | Máximo de iterações |

```bash
# CPU-only (7 workers, 0 CUDA)
STARPU_NCPUS=7 STARPU_NCUDA=0 \
mpirun -np 1 --bind-to none \
    ./kmeans_starpu input_100M_2d.txt 50 output/starpu_cpu 28 0 42 30

# GPU-only (0 CPU workers, 1 GPU, 1 chunk para manter dados na GPU)
STARPU_NCPUS=0 STARPU_NCUDA=1 \
mpirun -np 1 --bind-to none \
    ./kmeans_starpu input_100M_2d.txt 50 output/starpu_gpu 1 0 42 30

# Híbrido DMDA (7 CPU + 1 GPU, 32 chunks)
STARPU_NCPUS=7 STARPU_NCUDA=1 STARPU_SCHED=dmda \
mpirun -np 1 --bind-to none \
    ./kmeans_starpu input_100M_2d.txt 50 output/starpu_hybrid 32 0 42 30

# Multi-nó (np=4)
STARPU_NCPUS=7 STARPU_NCUDA=1 STARPU_SCHED=dmda \
mpirun -np 4 --hostfile hostfile --bind-to none \
    ./kmeans_starpu input_100M_2d.txt 50 output/starpu_np4 32 0 42 30
```

---

## Variáveis de Ambiente

### StarPU

| Variável | Exemplo | Descrição |
|----------|---------|-----------|
| `STARPU_NCPUS` | `7` | Número de workers CPU |
| `STARPU_NCUDA` | `1` | Número de GPUs CUDA |
| `STARPU_NOPENCL` | `0` | Desabilita OpenCL |
| `STARPU_SCHED` | `dmda` | Scheduler (`dmda`, `eager`, `lws`, ...) |
| `STARPU_FXT_PREFIX` | `logs/traces/run_` | Prefixo para arquivos de trace FXT |
| `STARPU_CALIBRATE` | `1` | Força recalibração dos modelos de performance |

### OpenMP

| Variável | Exemplo | Descrição |
|----------|---------|-----------|
| `OMP_NUM_THREADS` | `7` | Número de threads OpenMP por rank |

### Paths

| Variável | Valor padrão usado | Descrição |
|----------|-------------------|-----------|
| `LD_LIBRARY_PATH` | `${CUDA_LIB}:${STARPU_LIB}` | Bibliotecas dinâmicas |

---

## Scripts de Benchmark

Os scripts automatizam a execução de todos os cenários com logging estruturado.

### Scripts disponíveis

| Script | Dataset | np | Iterações | Logs |
|--------|---------|-----|-----------|------|
| `run_all.sh` | 100M | 1,2,4 | 30 | `logs/` |
| `run_np1.sh` | 100M | 1 | 30 | `logs/np1/` |
| `run_np2.sh` | 100M | 2 | 30 | `logs/np2/` |
| `run_np4.sh` | 100M | 4 | 30 | `logs/np4/` |
| `run_200M.sh` | 200M | 1 | 30 | `logs/np1_200M/` |
| `run_all_300.sh` | 300M | 1,2,4 | 30 | `logs300/` |
| `run_np1_300.sh` | 300M | 1 | 30 | `logs300/np1/` |
| `run_np2_300.sh` | 300M | 2 | 30 | `logs300/np2/` |
| `run_np4_300.sh` | 300M | 4 | 30 | `logs300/np4/` |

### Cenários executados por script (np=1)

| ID | Versão | Modo | Chunks | GPU Ratio |
|----|--------|------|--------|-----------|
| 0 | SEQ | — | — | — |
| 1 | OMP | CPU (`0`) | 1 | 0.5 |
| 2 | OMP | GPU (`1`) | 1 | 0.5 |
| 3 | OMP | Híbrido (`2`) | 2 | 0.70 |
| 4 | OMP | Híbrido (`2`) | 2 | 0.80 |
| 5 | OMP | Híbrido (`2`) | 2 | 0.90 |
| 6 | StarPU | CPU-only | 28 | — |
| 7 | StarPU | GPU-only | 8 | — |
| 8 | StarPU | Híbrido DMDA | 32 | — |

### Executando

```bash
# Dataset único, 1 nó
bash run_np1.sh

# Todos os nós com dataset customizado
DATASET=input_200M_2d.txt bash run_all.sh

# 300M pontos, todos os nós
bash run_all_300.sh
```

### Estrutura de logs

```
logs/
├── run_all.log          # Log geral
├── np1/
│   ├── run_np1.log      # Log do script
│   ├── seq_30iter.log
│   ├── omp_cpu_30iter.log
│   ├── omp_gpu_30iter.log
│   ├── starpu_hybrid_30iter.log
│   ├── starpu_hybrid_30iter_trace.log
│   └── traces/
│       └── np1_hybrid_30iter_*.fxt
├── np2/
└── np4/
```

---

## Formato dos Arquivos

### Entrada

```
<N> <D>
x1_1 x1_2 ... x1_D
x2_1 x2_2 ... x2_D
...
```

- Linha 1: número de pontos `N` e dimensões `D`
- Linhas seguintes: coordenadas separadas por espaço ou vírgula
- Tipo: `double` (ponto flutuante de 64 bits)

```
# Exemplo: 5 pontos, 2 dimensões
5 2
278.222556 -350.811353
-585.860899 240.580045
956.535763 220.095813
123.456789 -456.789012
-789.012345 678.901234
```

### Saída — `{OUT-DIR}/{K}-points.txt`

Um ID de cluster por linha (indexado em 1):

```
2
1
1
3
2
```

### Saída — `{OUT-DIR}/{K}-clusters.txt`

K linhas, uma por centroide (coordenadas separadas por espaço, 6 casas decimais):

```
-123.456789 234.567890
456.789012 -345.678901
789.012345 -678.901234
```

---

## Métricas Coletadas

Todas as versões imprimem métricas ao final da execução (stdout + arquivo de log).

### Sequencial (`SeqMetrics`)

| Campo | Descrição |
|-------|-----------|
| `loop_ms` | Tempo do loop K-Means (sem I/O) |
| `total_ms` | Tempo total (com I/O) |
| `iterations` | Iterações até convergência |
| `sse` | Soma dos Erros Quadráticos |

### OpenMP (`OmpMetrics`)

| Campo | Descrição |
|-------|-----------|
| `loop_ms` / `total_ms` | Tempos de execução |
| `iterations` | Iterações até convergência |
| `sse` | Soma dos Erros Quadráticos |
| `mpi_ranks` | Número de processos MPI |
| `omp_threads` | Threads OpenMP por rank |
| `mode` | Modo de execução (0/1/2) |
| `chunks` | Chunks por rank |
| `assign_count` / `calc_count` / `update_count` | Contagem de operações |

### StarPU (`StarPUMetrics`)

| Campo | Descrição |
|-------|-----------|
| `loop_ms` / `total_ms` | Tempos de execução |
| `iterations` | Iterações até convergência |
| `sse` | Soma dos Erros Quadráticos |
| `mpi_ranks` | Processos MPI |
| `cpu_workers` / `cuda_workers` | Workers CPU e GPU |
| `assign_cpu` / `assign_cuda` | Tarefas de atribuição por device |
| `calc_cpu` / `calc_cuda` | Tarefas de acumulação por device |
| `update_count` | Tarefas de atualização de centroides |

---

## Estrutura de Diretórios

```
KMeans-StarPU-Heterogeneous-Analysis/
├── Makefile                    # Build principal (StarPU, auto-detects CUDA)
├── generate_input.py           # Gerador de datasets paralelo
├── README.md
│
├── include/                    # Headers compartilhados
│   ├── kmeans_types.h          # Classes Point e Cluster
│   ├── metrics.h               # Structs de métricas (Seq, OMP, StarPU)
│   ├── options.h               # KMeansOptions
│   └── kmeans_mpi_tags.h       # Tags MPI para StarPU (evita colisões)
│
├── src/
│   ├── common/                 # Código compartilhado entre versões
│   │   ├── io.cpp              # Leitura de arquivos (read_points_from_file)
│   │   ├── metrics.cpp         # Impressão de métricas
│   │   └── metrics_simple.cpp  # Variante simplificada de métricas
│   │
│   ├── sequencial/
│   │   ├── Makefile
│   │   ├── kmeans.cpp          # Entry point (main)
│   │   ├── kmeans_seq.cpp      # Kernels: assign, accumulate, update
│   │   └── kmeans_seq.h
│   │
│   ├── openmp/
│   │   ├── Makefile            # Auto-detecta nvc++ (HPC SDK)
│   │   ├── kmeans_main.cpp     # Entry point (main + MPI init)
│   │   ├── kmeans_cpu.cpp      # Kernels CPU (OpenMP)
│   │   ├── kmeans_gpu.cpp      # Kernels GPU (OpenACC/CUDA)
│   │   └── kmeans_omp_mpi.h    # Declarações e function pointers
│   │
│   └── starpu/
│       ├── starpu.mk           # Flags, paths StarPU/CUDA
│       ├── kmeans_main.cpp     # Entry point (MPI + StarPU init)
│       ├── kmeans_mpi.cpp      # Classe KMeans: submissão de tarefas, DAG
│       ├── kmeans_cpu.cpp      # Codelets CPU
│       ├── kmeans_cuda.cu      # Kernels CUDA
│       └── kmeans_runtime.h   # Classe KMeans, codelets, modelos de perf.
│
├── build/                      # Objetos compilados (.o)
├── output_bench/               # Saídas dos experimentos (np=1,2,4)
├── output_bench300/            # Saídas dos experimentos 300M
├── logs/                       # Logs de experimentos 100M
├── logs300/                    # Logs de experimentos 300M
│
├── run_all.sh                  # Orquestrador: np=1,2,4 | 100M | 30 iter
├── run_np1.sh                  # np=1 | 100M | 30 iter
├── run_np2.sh                  # np=2 | 100M | 30 iter
├── run_np4.sh                  # np=4 | 100M | 30 iter
├── run_200M.sh                 # np=1 | 200M | 30 iter
├── run_all_300.sh              # Orquestrador: np=1,2,4 | 300M | 30 iter
├── run_np1_300.sh              # np=1 | 300M | 30 iter
├── run_np2_300.sh              # np=2 | 300M | 30 iter
└── run_np4_300.sh              # np=4 | 300M | 30 iter
```

---

## Configuração de Hardware Alvo

Os experimentos são projetados para simular uma instância **AWS g6.4xlarge**:

| Componente | Especificação |
|------------|---------------|
| CPU | AMD EPYC 7R13 @ 3.7 GHz, 8 cores (HT desabilitado) |
| GPU | NVIDIA L4 22 GB, 300 GB/s de bandwidth |
| RAM | 64 GB DDR5 |
| Rede | 25 Gbps (AWS Nitro ENA) |
| Workers CPU | 7 (1 reservado para StarPU/MPI overhead) |

### Simulação local

Para simular o ambiente sem HT nos scripts de benchmark:

```bash
taskset -c 0,2,4,6,8,10,12,14   # Pina em 8 cores físicos (sem SMT)
export OMP_NUM_THREADS=7
export STARPU_NCPUS=7
```

---

## Troubleshooting

### StarPU não detecta GPU

```bash
# Verificar detecção
STARPU_NCUDA=1 starpu_machine_display | grep -E "CUDA|GPU"

# Verificar biblioteca
ldconfig -p | grep libcudart
ls ${CUDA_LIB}/libcudart.so*
```

### SSE diverge entre np=1 e np>1

Causa conhecida e corrigida: não-ranqueado-0 recebia zeros em vez dos dados reais. A correção usa `MPI_Scatter` para distribuir pontos do rank 0 para todos os ranks antes do registro no StarPU, com ownership em blocos contíguos (`i / chunks_per_rank`) em vez de round-robin (`i % world_size`).

Diferença residual de ~1% em np=4 é esperada — floating-point non-determinism por tamanhos de chunk não-uniformes (sem bug).

### OMP Hybrid mais lento que OMP GPU

Comportamento esperado: o modo híbrido executa chunks de GPU e CPU **sequencialmente**, não em paralelo. Para datasets grandes, o tempo de CPU domina.

### Erro: `binary SEQ not found`

```bash
make -C src/sequencial
```

### Erro: `binary OMP not found`

```bash
make -C src/openmp GPU=1    # com GPU
make -C src/openmp           # sem GPU
```

### Erro: `binary StarPU not found`

```bash
make clean && make
```

### Erro de linking: `libstarpumpi-1.4 not found`

```bash
export LD_LIBRARY_PATH=/home/bridge/starpu_install/lib:${LD_LIBRARY_PATH}
ldconfig -p | grep starpumpi
```

---

## Referências

- [StarPU — Runtime System for Heterogeneous Multicore Architectures](https://starpu.gitlabpages.inria.fr/)
- [StarPU Handbook 1.4](https://files.inria.fr/starpu/doc/starpu.pdf)
- [NVIDIA L4 GPU Datasheet](https://www.nvidia.com/en-us/data-center/l4/)
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proceedings of the 5th Berkeley Symposium.
