![Clusters File Syntax](image/clusters.png)

# KMeans-StarPU-Heterogeneous-Analysis

## Ambiente e Configuração

### 1. Pré-requisitos no Host

- **Para CUDA (NVIDIA):**
  - Instale o driver NVIDIA mais recente para sua GPU.
  - Instale o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) para permitir acesso à GPU dentro do Docker.
  - Verifique se o comando abaixo mostra sua GPU:
    ```bash
    nvidia-smi
    ```

- **Para OpenCL (AMD/Intel/NVIDIA):**
  - Instale o driver da sua GPU (AMD, Intel ou NVIDIA).
  - Para AMD: [AMD ROCm](https://rocm.docs.amd.com/en/latest/) ou driver proprietário.
  - Para Intel: [Intel OpenCL Runtime](https://github.com/intel/compute-runtime).
  - Para NVIDIA: O driver já inclui suporte OpenCL.

---

### 2. Baixando a Imagem Docker

- **Para CUDA (NVIDIA):**
  ```bash
  docker pull nvidia/cuda:12.9.1-devel-ubuntu24.04
  ```

- **Para OpenCL (AMD/Intel):**
  - Use uma imagem Ubuntu padrão e instale o driver OpenCL dentro do container, ou utilize uma imagem ROCm para AMD.

- **Para ambas (recomendado para heterogeneidade):**
  - Use a imagem CUDA acima e instale o pacote OpenCL dentro do container.

---

### 3. Criando e Iniciando o Container

- **Com suporte a CUDA (NVIDIA):**
  ```bash
  docker run --gpus all -it \
    -v /CAMINHO/DO/SEU/PROJETO:/workspace \
    nvidia/cuda:12.9.1-devel-ubuntu24.04 \
    /bin/bash
  ```
  > Substitua `/CAMINHO/DO/SEU/PROJETO` pelo caminho real do seu projeto no host.

- **Com suporte a OpenCL (AMD/Intel):**
  ```bash
  docker run -it \
    -v /CAMINHO/DO/SEU/PROJETO:/workspace \
    ubuntu:24.04 \
    /bin/bash
  ```
  > Dentro do container, instale o driver OpenCL e o ICD correspondente ao seu hardware.

---

### 4. Instalando Dependências no Container

```bash
apt-get update
apt-get install -y build-essential git ocl-icd-opencl-dev libhwloc-dev wget pkg-config vim
```

- **Para CUDA:** (já incluso na imagem CUDA)
- **Para OpenCL:** O comando acima instala o ICD genérico. Para AMD/Intel, consulte a documentação do fabricante para instalar o runtime específico.

---

### 5. Instalando a StarPU (se necessário)

```bash
wget https://files.inria.fr/starpu/starpu-1.4.0/starpu-1.4.0.tar.gz
tar xzf starpu-1.4.0.tar.gz
cd starpu-1.4.0
./configure --enable-cuda --enable-opencl
make -j$(nproc)
make install
cd ..
```

---

### 6. Configurando Variáveis de Ambiente

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

---

### 7. Acesse o diretório do projeto

```bash
cd /workspace
```

---

## Compilação

### 1. Apenas CPU

```bash
g++ -O3 kmeans_starpu.cpp -o kmeans_starpu_cpu \
    -I./starpu/include \
    -L./starpu/src/.libs \
    -lstarpu-1.4 \
    -lpthread -lm
```

---

### 2. CPU + CUDA

```bash
# Compile o kernel CUDA
nvcc -I/usr/local/include/starpu/1.4 -I/usr/local/include -c assign_point_cuda.cu -o assign_point_cuda.o

# Linke tudo
g++ -O3 kmeans_starpu.cpp assign_point_cuda.o -o kmeans_starpu_cuda \
    -I./starpu/include -I/usr/local/cuda/include \
    -L./starpu/src/.libs -L/usr/local/cuda/lib64 \
    -lstarpu-1.4 -lcuda -lcudart -lpthread -lm
```

---

### 3. CPU + CUDA + OpenCL

```bash
# Compile o kernel CUDA
nvcc -I/usr/local/include/starpu/1.4 -I/usr/local/include -c assign_point_cuda.cu -o assign_point_cuda.o

# Compile o kernel OpenCL
gcc -I/usr/local/include/starpu/1.4 -I/usr/local/include -I/usr/local/cuda/include -c assign_point_opencl.c -o assign_point_opencl.o

# Linke tudo
g++ -O3 -DSTARPU_USE_CUDA -DSTARPU_USE_OPENCL \
    kmeans_starpu.cpp assign_point_cuda.o assign_point_opencl.o -o kmeans_starpu_cuda_opencl \
    -I./starpu/include -I/usr/local/cuda/include \
    -L./starpu/src/.libs -L/usr/local/cuda/lib64 \
    -lstarpu-1.4 -lcuda -lcudart -lOpenCL -lpthread -lm
```

---

> **Observações:**
> - Se não quiser OpenCL, basta não compilar/linkar `assign_point_opencl.c` e não usar `-DSTARPU_USE_OPENCL`.
> - Se não quiser CUDA, basta não compilar/linkar `assign_point_cuda.cu` e não usar `-DSTARPU_USE_CUDA`.
> - Ajuste os caminhos de include/lib se necessário conforme seu ambiente.
> - Se aparecer erro de referência a OpenCL, remova todas as referências a `assign_point_to_cluster_opencl` e `get_opencl_kernel_calls` do código.

---

## Execução

Antes de rodar, garanta que as bibliotecas estejam no `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=./starpu/src/.libs:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Execute o programa conforme o binário gerado, por exemplo:

```bash
./kmeans_starpu_cuda_opencl <INPUT> <K> <OUT-DIR>
```

---

Pronto! Agora seu ambiente está preparado para rodar em máquinas com GPU NVIDIA (CUDA), OpenCL (AMD/Intel/NVIDIA) ou ambas,