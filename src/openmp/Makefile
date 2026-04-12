# Força o MPI a usar o G++ 13 (Moderno) em vez do G++ 11 padrão do seu sistema
ifeq ($(GPU), 1)
    export OMPI_CXX=g++-13
endif

CXX = mpicxx

# Por padrão, compila sem GPU. Para usar GPU: make GPU=1
GPU ?= 0

# ==============================================================================
# FLAGS DE PERFORMANCE EXTREMA (CPU)
# -O3: Nível máximo de otimização
# -march=native -mtune=native: Destrava AVX/AVX2/AVX-512 (Vetorização)
# -ffast-math: Ativa FMA (Fused Multiply-Add) para matemática ultrarrápida
# -DNDEBUG: Remove asserções de Debug do código C++
# ==============================================================================
CXXFLAGS = -O3 -march=native -mtune=native -ffast-math -DNDEBUG -Wall -fopenmp -fcf-protection=none -fno-stack-protector -I../../include
LDFLAGS = -fopenmp

# Lista base de objetos (Main e CPU)
OBJS = kmeans_main.o kmeans_cpu.o
IO_OBJ = ../../build/io.o

# --- A CHAVE DE ESCALABILIDADE ---
ifeq ($(GPU), 1)
    $(info >>> Compilando COM suporte a GPU via GCC 13 (Offload ativado))
    
    CXXFLAGS += -DUSE_GPU
    
    # ==============================================================================
    # FLAGS DE PERFORMANCE EXTREMA (GPU OFFLOAD)
    # Adicionamos -O3 e -ffast-math DENTRO da string do PTX para garantir
    # que o Assembly da placa de vídeo seja agressivamente otimizado.
    # ==============================================================================
    OFFLOAD_FLAGS = -foffload=nvptx-none="-O3 -ffast-math -misa=sm_70 -fcf-protection=none -fno-stack-protector"
    
    CXXFLAGS += $(OFFLOAD_FLAGS)
    LDFLAGS += $(OFFLOAD_FLAGS)
    
    # Adiciona o arquivo da GPU na lista de compilação
    OBJS += kmeans_gpu.o
else
    $(info >>> Compilando SEM suporte a GPU (100% CPU))
    LDFLAGS += -foffload=disable
endif

all: kmeans_openmp

kmeans_main.o: kmeans_main.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c kmeans_main.cpp -o kmeans_main.o

kmeans_cpu.o: kmeans_cpu.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c kmeans_cpu.cpp -o kmeans_cpu.o

kmeans_gpu.o: kmeans_gpu.cpp kmeans_omp_mpi.h
	$(CXX) $(CXXFLAGS) -c kmeans_gpu.cpp -o kmeans_gpu.o

kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(IO_OBJ) $(LDFLAGS) -o kmeans_openmp

clean:
	rm -f *.o kmeans_openmp