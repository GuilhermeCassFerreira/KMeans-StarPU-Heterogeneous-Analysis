# Sub-Makefile com flags específicas do StarPU e NVCC

NVCC = nvcc
CXX = mpic++

# 1. Flags e caminhos BASE (Apenas CPU e StarPU)
CXXFLAGS = -O3 -std=c++11 -fPIC -DUSE_MPI
NVCCFLAGS = -ccbin /usr/bin/g++-11 -std=c++11 -Xcompiler "-fPIC" -O3 -DSTARPU_USE_CUDA -gencode arch=compute_86,code=sm_86

STARPU_PATH = /home/bridge/starpu_install
CUDA_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1

INCLUDES = -I$(STARPU_PATH)/include/starpu/1.4
LDFLAGS = -L$(STARPU_PATH)/lib
LDLIBS = -lstarpumpi-1.4 -lstarpu-1.4 -lpthread -lm

# 2. Injeção Condicional (Ativada apenas se o Makefile principal disser USE_CUDA=1)
ifeq ($(USE_CUDA), 1)
    CUDA_MATH_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/math_libs/13.1/targets/x86_64-linux
    CXXFLAGS += -DSTARPU_USE_CUDA
    INCLUDES += -I$(CUDA_PATH)/targets/x86_64-linux/include -I$(CUDA_MATH_PATH)/include
    LDFLAGS += -L$(CUDA_PATH)/lib64 -L/usr/local/cuda/lib64 -L$(CUDA_MATH_PATH)/lib
    LDLIBS += -lcuda -lcudart
endif