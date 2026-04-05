CXX = mpicxx

# Por padrão, compila sem GPU para você testar agora
GPU ?= 0

# Flags básicas de CPU
CXXFLAGS = -O3 -Wall -fopenmp -fcf-protection=none -I../../include -g
LDFLAGS = -fopenmp

# Lista base de objetos (Main e CPU)
OBJS = kmeans_main.o kmeans_cpu.o
IO_OBJ = ../../build/io.o

# --- A CHAVE DE ESCALABILIDADE ---
ifeq ($(GPU), 1)
    $(info >>> Compilando COM suporte a GPU (Offload ativado))
    # Adiciona a flag no C++ para ativar os #ifdefs
    CXXFLAGS += -DUSE_GPU
    # Adiciona as flags insanas que o seu driver da NVIDIA exige
    # (Quando você consertar o GCC ou for pro Clang, você mexe nessas flags)
    CXXFLAGS += -foffload=nvptx-none="-misa=sm_70 -fcf-protection=none"
    LDFLAGS += -foffload=nvptx-none="-misa=sm_70 -fcf-protection=none"
    # Adiciona o arquivo da GPU na lista de compilação
    OBJS += kmeans_gpu.o
else
    $(info >>> Compilando SEM suporte a GPU (100% CPU))
    LDFLAGS += -foffload=disable
endif

all: kmeans_openmp

kmeans_main.o: kmeans_main.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_main.cpp -o kmeans_main.o

kmeans_cpu.o: kmeans_cpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_cpu.cpp -o kmeans_cpu.o

kmeans_gpu.o: kmeans_gpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_gpu.cpp -o kmeans_gpu.o

kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(IO_OBJ) $(LDFLAGS) -o kmeans_openmp

clean:
	rm -f *.o kmeans_openmp