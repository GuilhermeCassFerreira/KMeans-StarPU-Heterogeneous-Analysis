# ==============================================================================
# CONFIGURAÇÃO DE AMBIENTE - BRIDGE LABS / UFSC
# ==============================================================================

# Caminhos baseados no seu comando 'find' e ambiente do HPC SDK
HPC_SDK_PATH := /opt/nvidia/hpc_sdk/Linux_x86_64/26.3
CUDA_VER     := 13.1
CUDA_HOME    := $(HPC_SDK_PATH)/cuda/$(CUDA_VER)

# Exportações CRÍTICAS para o compilador NVIDIA encontrar o ptxas e as libs
export NVHPC_CUDA_HOME := $(CUDA_HOME)
export PATH            := $(CUDA_HOME)/bin:$(HPC_SDK_PATH)/compilers/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)
export OMPI_CXX        := $(HPC_SDK_PATH)/compilers/bin/nvc++

# Compilador (mpicxx usará o nvc++ devido ao OMPI_CXX acima)
CXX = mpicxx
GPU ?= 0

# Flags de Performance:
# -fast: Otimização máxima (vetorização, inlining, etc)
# -Minfo=mp: Relatório de paralelismo (importante para ver o offload)
CXXFLAGS = -fast -DNDEBUG -Wall -I../../include
LDFLAGS  = 

# Objetos do projeto
OBJS = kmeans_main.o kmeans_cpu.o
IO_OBJ = ../../build/io.o

# Configuração para compilação com suporte a GPU (OpenMP Offload)
ifeq ($(GPU), 1)
    $(info >>> COMPILANDO MODO MULTI-GPU (NVC++ / cc86) <<<)
    # cc86 é a arquitetura da RTX 2050/3060/4000
    CXXFLAGS += -DUSE_GPU -mp=gpu -gpu=cc86 -Minfo=mp
    
    # ATENÇÃO: Adicionado -L e -lcudart para encontrar o cudaHostRegister
    LDFLAGS  += -mp=gpu -gpu=cc86 -L$(CUDA_HOME)/lib64 -lcudart
    
    OBJS     += kmeans_gpu.o
else
    $(info >>> COMPILANDO MODO CPU (OpenMP / AVX2) <<<)
    CXXFLAGS += -mp
    LDFLAGS  += -mp
endif

# Alvo principal
all: kmeans_openmp

# Regra de compilação dos objetos
%.o: %.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Linkagem final
kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(IO_OBJ) $(LDFLAGS) -o kmeans_openmp

# Limpeza
clean:
	rm -f *.o kmeans_openmp