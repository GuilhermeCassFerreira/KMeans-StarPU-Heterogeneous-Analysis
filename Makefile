# =============================================================================
# Makefile Principal - KMeans StarPU
# =============================================================================

NVCC_PATH := $(shell command -v nvcc 2> /dev/null)

ifdef NVCC_PATH
    $(info ===> [AUTO-CONFIG] GPU CUDA detectada. Modo Hibrido.)
    USE_CUDA = 1
else
    $(info ===> [AUTO-CONFIG] Sem CUDA. Modo CPU.)
    USE_CUDA = 0
endif

include src/starpu/starpu.mk

INCLUDE_DIR = include
SRC_COMMON  = src/common
SRC_STARPU  = src/starpu

PROJECT_INCLUDES = -I$(INCLUDE_DIR) -I$(SRC_STARPU) $(INCLUDES)

BUILD_DIR = build

OBJS = $(BUILD_DIR)/io.o \
       $(BUILD_DIR)/metrics.o \
       $(BUILD_DIR)/kmeans_cpu.o \
       $(BUILD_DIR)/kmeans_mpi.o \
       $(BUILD_DIR)/kmeans_main.o

ifeq ($(USE_CUDA), 1)
    OBJS += $(BUILD_DIR)/kmeans_cuda.o
endif

TARGET = kmeans_starpu

# =============================================================================
# Build
# =============================================================================

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
ifeq ($(USE_CUDA), 1)
	$(NVCC) -ccbin mpic++ $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)
else
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)
endif

# =============================================================================
# Compilação
# =============================================================================

$(BUILD_DIR)/io.o: $(SRC_COMMON)/io.cpp
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/metrics.o: $(SRC_COMMON)/metrics.cpp
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_cpu.o: $(SRC_STARPU)/kmeans_cpu.cpp
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_mpi.o: $(SRC_STARPU)/kmeans_mpi.cpp
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_main.o: $(SRC_STARPU)/kmeans_main.cpp
	$(CXX) $(CXXFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

$(BUILD_DIR)/kmeans_cuda.o: $(SRC_STARPU)/kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(PROJECT_INCLUDES) -c $< -o $@

# =============================================================================
# Clean
# =============================================================================

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean