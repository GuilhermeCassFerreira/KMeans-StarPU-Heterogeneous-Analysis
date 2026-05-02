# =============================================================================
# Sub-Makefile com flags específicas do StarPU e NVCC (HPC SDK Version)
# =============================================================================

NVCC = nvcc
CXX = mpic++

# Flags BASE
CXXFLAGS = -O3 -std=c++11 -fPIC
NVCCFLAGS = -ccbin mpic++ -std=c++11 -Xcompiler "-fPIC" -O3 -DSTARPU_USE_CUDA \
            -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

STARPU_PATH = /home/bridge/starpu_install
CUDA_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1
MATH_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/26.3/math_libs/13.1/targets/x86_64-linux

INCLUDES = -I$(STARPU_PATH)/include/starpu/1.4
LDLIBS   = -lstarpumpi-1.4 -lstarpu-1.4 -lpthread -lm

# 🛠️ AQUI ESTÁ O TRUQUE: nvcc precisa de -Xlinker para o rpath
LDFLAGS  = -L$(STARPU_PATH)/lib -Xlinker -rpath -Xlinker $(STARPU_PATH)/lib

ifeq ($(USE_CUDA), 1)
    CXXFLAGS += -DSTARPU_USE_CUDA
    INCLUDES += -I$(CUDA_PATH)/include -I$(MATH_PATH)/include
    LDFLAGS  += -L$(CUDA_PATH)/targets/x86_64-linux/lib -L$(MATH_PATH)/lib
    LDLIBS   += -lcuda -lcudart
endif