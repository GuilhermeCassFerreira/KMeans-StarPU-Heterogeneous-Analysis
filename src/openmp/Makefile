# ==============================================================================
# CONFIGURAÇÃO DE AMBIENTE - BRIDGE LABS / UFSC (AUTO-DETECÇÃO)
# ==============================================================================

# Caminho base do HPC SDK
HPC_SDK_PATH := /opt/nvidia/hpc_sdk/Linux_x86_64/26.3
NVC_COMPILER := $(HPC_SDK_PATH)/compilers/bin/nvc++

# 1. AUTO-DETECÇÃO: Verifica se o compilador nvc++ existe na máquina
HAS_NVHPC := $(shell if [ -f "$(NVC_COMPILER)" ]; then echo 1; else echo 0; fi)

CXX = mpicxx
GPU ?= 0

# Objetos base (Sempre compilados)
OBJS = kmeans_main.o kmeans_cpu.o
IO_OBJ    = ../../build/io.o
METRICS_OBJ = ../../build/metrics.o

# ==============================================================================
# BLOCO 1: TEM NVIDIA HPC SDK INSTALADO
# ==============================================================================
ifeq ($(HAS_NVHPC), 1)
    $(info ===> [AUTO-CONFIG] NVIDIA HPC SDK detectado. Usando nvc++.)

    CUDA_VER     := 13.1
    CUDA_HOME    := $(HPC_SDK_PATH)/cuda/$(CUDA_VER)

    # Exportações CRÍTICAS para o compilador NVIDIA
    export NVHPC_CUDA_HOME := $(CUDA_HOME)
    export PATH            := $(CUDA_HOME)/bin:$(HPC_SDK_PATH)/compilers/bin:$(PATH)
    export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)
    export OMPI_CXX        := $(NVC_COMPILER)

    # Flags base do NVC++
    CXXFLAGS = -O3 -march=native -DNDEBUG -Wall -I../../include -I../common -DUSE_MPI
    LDFLAGS  =

    ifeq ($(GPU), 1)
        $(info >>> COMPILANDO MODO MULTI-GPU (NVC++ / cc86) <<<)
        CXXFLAGS += -DUSE_GPU -mp=gpu -gpu=cc86 -Minfo=mp
        LDFLAGS  += -mp=gpu -gpu=cc86 -L$(CUDA_HOME)/lib64 -lcudart
        OBJS     += kmeans_gpu.o
    else
        $(info >>> COMPILANDO MODO CPU (NVC++ OpenMP / AVX2) <<<)
        CXXFLAGS += -mp
        LDFLAGS  += -mp
    endif

# ==============================================================================
# BLOCO 2: NÃO TEM NVIDIA INSTALADO (FALLBACK PARA GCC)
# ==============================================================================
else
    $(info ===> [AUTO-CONFIG] NVIDIA HPC SDK NAO detectado. Fazendo fallback para GCC/g++.)
    
    # Força o MPI a usar o compilador C++ padrão do sistema
    export OMPI_CXX := g++

    # Se o usuário tentar forçar GPU=1 sem ter o compilador, nós ignoramos
    ifeq ($(GPU), 1)
        $(warning !!! GPU=1 ignorado pois o toolkit da NVIDIA nao esta presente !!!)
    endif

    $(info >>> COMPILANDO MODO EXCLUSIVO CPU (GCC / OpenMP) <<<)
    
    # Flags equivalentes para o G++
    CXXFLAGS = -O3 -DNDEBUG -Wall -I../../include -I../common -fopenmp -DUSE_MPI
    LDFLAGS  = -fopenmp
endif

# ==============================================================================
# REGRAS DE COMPILAÇÃO
# ==============================================================================

all: kmeans_openmp

# Regra de compilação dos objetos do diretório local
%.o: %.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Linkagem final
kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(IO_OBJ) $(METRICS_OBJ) $(LDFLAGS) -o kmeans_openmp

# Limpeza
clean:
	rm -f *.o kmeans_openmp

.PHONY: all clean