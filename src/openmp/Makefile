CXX = mpic++
# Forçamos o wrapper a usar g++ para evitar erro de 'clang++ not found'
export OMPI_CXX = g++

# Flags para CPU multi-core
CXXFLAGS = -O3 -Wall -fopenmp -I../../include -g
LDFLAGS = -fopenmp

# Caminho para o objeto IO gerado pelo build do StarPU na raiz
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