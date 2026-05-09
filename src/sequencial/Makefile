CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -I../../include -I../common
LDFLAGS = -lm

COMMON_OBJ = ../common/io.o ../common/metrics_simple.o
SEQ_OBJ = kmeans.o kmeans_seq.o
SEQ_TARGET = kmeans_seq

all: $(SEQ_TARGET)

$(SEQ_TARGET): $(SEQ_OBJ) $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../common/%.o: ../common/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(SEQ_OBJ) $(COMMON_OBJ) $(SEQ_TARGET)

.PHONY: all clean