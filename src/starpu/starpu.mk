# Sub-Makefile com flags específicas do StarPU e NVCC

NVCC = nvcc
CXX = mpic++

# 1. Flags e caminhos BASE (Apenas CPU e StarPU)
CXXFLAGS = -O3 -std=c++11 -fPIC
NVCCFLAGS = -ccbin /usr/bin/g++-11 -std=c++11 -Xcompiler "-fPIC" -O3 -DSTARPU_USE_CUDA

STARPU_PATH = /home/bridge/starpu_install
CUDA_PATH = /usr/local/cuda

INCLUDES = -I$(STARPU_PATH)/include/starpu/1.4
LDFLAGS = -L$(STARPU_PATH)/lib
LDLIBS = -lstarpumpi-1.4 -lstarpu-1.4 -lpthread -lm

# 2. Injeção Condicional (Ativada apenas se o Makefile principal disser USE_CUDA=1)
ifeq ($(USE_CUDA), 1)
    CXXFLAGS += -DSTARPU_USE_CUDA
    INCLUDES += -I$(CUDA_PATH)/include
    LDFLAGS += -L$(CUDA_PATH)/lib64
    LDLIBS += -lcuda -lcudart
endif