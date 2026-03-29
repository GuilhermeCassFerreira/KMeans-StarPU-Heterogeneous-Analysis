CXX = mpic++
CXXFLAGS = -O3 -std=c++11 -Wall

# Removemos o metrics.cpp daqui
COMMON_SRC = ../common/io.cpp
SEQ_SRC = kmeans_cpu.cpp kmeans_mpi.cpp kmeans_main.cpp

all: kmeans_seq

kmeans_seq: $(COMMON_SRC) $(SEQ_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f kmeans_seq