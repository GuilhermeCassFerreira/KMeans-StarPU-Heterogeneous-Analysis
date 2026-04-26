CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -I../../include -I../common
LDFLAGS = -lm

# Arquivos comuns (relativos ao diretório atual)
COMMON_SRC = ../common/io.cpp
COMMON_OBJ = ../common/io.o

# Sequencial
SEQ_SRC = kmeans_main.cpp kmeans_seq_kernels.cpp
SEQ_OBJ = kmeans_main.o kmeans_seq_kernels.o
SEQ_TARGET = kmeans_seq

# Targets
all: $(SEQ_TARGET)

$(SEQ_TARGET): $(SEQ_OBJ) $(COMMON_OBJ)
$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $< -o $@

clean:
    rm -f $(SEQ_OBJ) $(COMMON_OBJ) $(SEQ_TARGET)

.PHONY: all clean