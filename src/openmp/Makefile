# ==============================================================================
# CONFIGURAÇÃO DE AMBIENTE - BRIDGE LABS
# ==============================================================================
# Apontamos para o binário do compilador diretamente
NVC_BIN = /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nvc++

# Garantimos que o PATH do sistema e do CUDA estejam visíveis para o MPI
export PATH := /usr/local/cuda/bin:/usr/bin:/bin:$(PATH)
export OMPI_CXX = $(NVC_BIN)
export NVHPC_CUDA_HOME = /usr/local/cuda

CXX = mpicxx
GPU ?= 0

# Flags de Performance NVC++
CXXFLAGS = -fast -DNDEBUG -Wall -I../../include
LDFLAGS = 

OBJS = kmeans_main.o kmeans_cpu.o
IO_OBJ = ../../build/io.o

ifeq ($(GPU), 1)
    $(info >>> Compilando MODO GPU (Redução em Shared Memory) <<<)
    CXXFLAGS += -DUSE_GPU
    # cc86 é a sua RTX 2050 (Ampere)
    OFFLOAD_FLAGS = -mp=gpu -gpu=cc86 -Minfo=mp
    CXXFLAGS += $(OFFLOAD_FLAGS)
    LDFLAGS += $(OFFLOAD_FLAGS)
    OBJS += kmeans_gpu.o
else
    $(info >>> Compilando MODO CPU (Vetorização AVX2) <<<)
    CXXFLAGS += -mp
    LDFLAGS += -mp
endif

all: kmeans_openmp

# Regra genérica para objetos
%.o: %.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(IO_OBJ) $(LDFLAGS) -o kmeans_openmp

clean:
	rm -f *.o kmeans_openmp