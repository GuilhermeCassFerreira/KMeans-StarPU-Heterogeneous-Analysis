# ==============================================================================
# CONFIGURAÇÃO DE AMBIENTE - BRIDGE LABS / UFSC
# ==============================================================================
# Caminhos para o CUDA e Compilador NVIDIA
export NVHPC_CUDA_HOME := /usr/local/cuda
export PATH := /usr/local/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$(PATH)
export OMPI_CXX := /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nvc++

CXX = mpicxx
GPU ?= 0

# Flags de Performance:
# -fast: Otimização máxima para i7 (AVX2) e NVIDIA.
# -Minfo=mp: Relatório de paralelismo.
CXXFLAGS = -fast -DNDEBUG -Wall -I../../include
LDFLAGS = 

OBJS = kmeans_main.o kmeans_cpu.o
IO_OBJ = ../../build/io.o

ifeq ($(GPU), 1)
    $(info >>> COMPILANDO MODO MULTI-GPU (NVC++) <<<)
    CXXFLAGS += -DUSE_GPU -mp=gpu -gpu=cc86 -Minfo=mp
    LDFLAGS += -mp=gpu -gpu=cc86
    OBJS += kmeans_gpu.o
else
    $(info >>> COMPILANDO MODO CPU (Vetorização AVX2) <<<)
    CXXFLAGS += -mp
    LDFLAGS += -mp
endif

all: kmeans_openmp

%.o: %.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(IO_OBJ) $(LDFLAGS) -o kmeans_openmp

clean:
	rm -f *.o kmeans_openmp