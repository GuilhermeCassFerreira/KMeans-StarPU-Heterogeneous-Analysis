# =============================================================================
# Makefile Principal - KMeans StarPU Heterogeneous
# =============================================================================

# Incluir configurações de compilação do StarPU
include src/starpu/starpu.mk

# Diretórios
INCLUDE_DIR = include
SRC_COMMON  = src/common
SRC_STARPU  = src/starpu

# Includes do projeto (headers locais + StarPU/CUDA)
PROJECT_INCLUDES = -I$(INCLUDE_DIR) -I$(SRC_STARPU) $(INCLUDES)

# Diretório de saída dos objetos
BUILD_DIR = build

# Objetos
OBJS = $(BUILD_DIR)/io.o \
	   $(BUILD_DIR)/metrics.o \
	   $(BUILD_DIR)/kmeans_cpu.o \
	   $(BUILD_DIR)/kmeans_cuda.o \
	   $(BUILD_DIR)/kmeans_mpi.o \
	   $(BUILD_DIR)/kmeans_main.o

# Alvo final
TARGET = kmeans_starpu

# =============================================================================
# Regras
# =============================================================================

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

# --- Código comum (independente do StarPU) ---

$(BUILD_DIR)/io.o: $(SRC_COMMON)/io.cpp $(INCLUDE_DIR)/kmeans_types.h
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/metrics.o: $(SRC_COMMON)/metrics.cpp
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

# --- Código StarPU ---

$(BUILD_DIR)/kmeans_cpu.o: $(SRC_STARPU)/kmeans_cpu.cpp $(SRC_STARPU)/kmeans_runtime.h $(INCLUDE_DIR)/kmeans_types.h
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_cuda.o: $(SRC_STARPU)/kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_mpi.o: $(SRC_STARPU)/kmeans_mpi.cpp $(SRC_STARPU)/kmeans_runtime.h $(INCLUDE_DIR)/kmeans_types.h $(INCLUDE_DIR)/options.h
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_main.o: $(SRC_STARPU)/kmeans_main.cpp $(SRC_STARPU)/kmeans_runtime.h $(INCLUDE_DIR)/kmeans_types.h $(INCLUDE_DIR)/options.h
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

# --- Limpeza ---

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean