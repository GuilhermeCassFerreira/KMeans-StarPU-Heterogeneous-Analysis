FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Atualizar pacotes
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libblas-dev \
    liblapack-dev \
    libhwloc-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Baixar e compilar o StarPU
RUN git clone https://gitlab.inria.fr/starpu/starpu.git /opt/starpu && \
    cd /opt/starpu && \
    ./configure --enable-cuda --prefix=/usr/local && \
    make -j$(nproc) && \
    make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
