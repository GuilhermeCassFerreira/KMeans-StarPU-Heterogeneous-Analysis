# Sub-Makefile com flags específicas do StarPU e NVCC

NVCC = nvcc
CXX = mpic++
CXXFLAGS = -O3 -std=c++11 -fPIC -DSTARPU_USE_CUDA
NVCCFLAGS = -ccbin /usr/bin/g++-11 -std=c++11 -Xcompiler "-fPIC" -O3 -DSTARPU_USE_CUDA

STARPU_PATH = /home/bridge/starpu_install
CUDA_PATH = /usr/local/cuda

INCLUDES = -I$(STARPU_PATH)/include/starpu/1.4 -I$(CUDA_PATH)/include
LDFLAGS = -L$(STARPU_PATH)/lib -L$(CUDA_PATH)/lib64
LDLIBS = -lstarpumpi-1.4 -lstarpu-1.4 -lcuda -lcudart -lpthread -lm