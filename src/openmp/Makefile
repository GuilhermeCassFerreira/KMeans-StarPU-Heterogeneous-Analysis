# O segredo é sobrescrever a variável que o OpenMPI usa internamente
CXX = mpicxx
export OMPI_CXX = g++

# Flags padrão de CPU (funcionam em qualquer x86_64)
CXXFLAGS = -O3 -Wall -fopenmp -I../../include -g
LDFLAGS = -fopenmp

# Caminho do objeto IO (ajuste se o StarPU salvar em outro lugar)
# Se estiver no build da raiz: ../../build/io.o
IO_OBJ = ../../build/io.o

all: kmeans_openmp

kmeans_main.o: kmeans_main.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_main.cpp -o kmeans_main.o

kmeans_cpu.o: kmeans_cpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_cpu.cpp -o kmeans_cpu.o

kmeans_gpu.o: kmeans_gpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_gpu.cpp -o kmeans_gpu.o

kmeans_openmp: kmeans_main.o kmeans_cpu.o kmeans_gpu.o
	$(CXX) $(CXXFLAGS) -o kmeans_openmp kmeans_main.o kmeans_cpu.o kmeans_gpu.o $(IO_OBJ) $(LDFLAGS)

clean:
	rm -f *.o kmeans_openmp