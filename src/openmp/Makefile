CXX = mpicxx
# Usando o clang++ para manter a consistência, mas sem offload
export OMPI_CXX = clang++

# Flags de Compilação: 
# Removemos o --offload-arch e as pastas do CUDA
# Mantemos o -fopenmp para o paralelismo de threads na CPU
CXXFLAGS = -O3 -Wall -fopenmp -I../../include -g

# Flags de Linkagem:
# Removemos as libs da NVIDIA e o omptarget
LDFLAGS = -fopenmp

all: kmeans_openmp

# Note que vamos compilar o kmeans_gpu.o também, 
# mas como ele está sem as flags de offload, o compilador vai ignorar os #pragma omp target
# ou você pode simplesmente não linká-lo se preferir.

kmeans_main.o: kmeans_main.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_main.cpp -o kmeans_main.o

kmeans_cpu.o: kmeans_cpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_cpu.cpp -o kmeans_cpu.o

kmeans_gpu.o: kmeans_gpu.cpp
	$(CXX) $(CXXFLAGS) -c kmeans_gpu.cpp -o kmeans_gpu.o

kmeans_openmp: kmeans_main.o kmeans_cpu.o kmeans_gpu.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o kmeans_openmp kmeans_main.o kmeans_cpu.o kmeans_gpu.o ../common/io.o

clean:
	rm -f *.o kmeans_openmp