# Compilador MPI
CXX = mpicxx

# Forçamos o uso do g++ ou clang++ via MPI sem as flags de GPU
# Adicionamos -fNo-stack-protector e -fcf-protection=none para evitar o erro do mkoffload
COMMON_FLAGS = -O3 -Wall -fopenmp -fcf-protection=none -I../../include -g

# Flags de Compilação
CXXFLAGS = $(COMMON_FLAGS)

# Flags de Linkagem (Removendo tudo que é GPU/Offload)
# O -foffload=disable é o segredo para o GCC ignorar a GPU e não dar erro de mkoffload
LDFLAGS = -fopenmp -foffload=disable

# Objetos
OBJS = kmeans_main.o kmeans_cpu.o kmeans_gpu.o ../../build/io.o

all: kmeans_openmp

kmeans_main.o: kmeans_main.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_main.cpp -o kmeans_main.o

kmeans_cpu.o: kmeans_cpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_cpu.cpp -o kmeans_cpu.o

kmeans_gpu.o: kmeans_gpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_gpu.cpp -o kmeans_gpu.o

kmeans_openmp: $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o kmeans_openmp

clean:
	rm -f *.o kmeans_openmp