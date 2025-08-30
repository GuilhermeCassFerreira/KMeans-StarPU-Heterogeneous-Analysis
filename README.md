![Clusters File Syntax](image/clusters.png)

# KMeans-StarPU-Heterogeneous-Analysis

## Ambiente e Configuração

### 1. Pré-requisitos no Host

- **Para CUDA (NVIDIA):**
  - Instale o driver NVIDIA mais recente para sua GPU.
  - Verifique se o comando abaixo mostra sua GPU:
    ```bash
    nvidia-smi
    ```
  - Instale o CUDA Toolkit compatível com seu driver.
  - Certifique-se de que o CUDA está no PATH e LD_LIBRARY_PATH:
    ```bash
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ```

- **Para OpenCL (AMD/Intel/NVIDIA):**
  - Instale o driver da sua GPU (AMD, Intel ou NVIDIA).
  - Para AMD: [AMD ROCm](https://rocm.docs.amd.com/en/latest/) ou driver proprietário.
  - Para Intel: [Intel OpenCL Runtime](https://github.com/intel/compute-runtime).
  - Para NVIDIA: O driver já inclui suporte OpenCL.

- **Para MPI:**
  - Instale o OpenMPI ou MPICH:
    ```bash
    sudo apt-get install -y libopenmpi-dev openmpi-bin
    ```

---

### 2. (Opcional) Ambiente Docker

- **Para CUDA (NVIDIA):**
  ```bash
  docker pull nvidia/cuda:12.9.1-devel-ubuntu24.04
  ```
  > Use `--gpus all` ao rodar o container para expor a GPU.

- **Para OpenCL (AMD/Intel):**
  - Use uma imagem Ubuntu padrão e instale o driver OpenCL dentro do container, ou utilize uma imagem ROCm para AMD.

---

### 3. Instalando Dependências no Host/Container

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ocl-icd-opencl-dev libhwloc-dev wget pkg-config vim
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

---

### 4. Instalando a StarPU (se necessário)

```bash
wget https://files.inria.fr/starpu/starpu-1.4.0/starpu-1.4.0.tar.gz
tar xzf starpu-1.4.0.tar.gz
cd starpu-1.4.0
./configure --enable-cuda --enable-opencl
make -j$(nproc)
sudo make install
cd ..
```

---

### 5. Configurando Variáveis de Ambiente

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export STARPU_NCUDA=1
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
nvcc -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/include -I/usr/local/cuda/include -c assign_point_cuda.cu -o assign_point_cuda.o

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
nvcc -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/include -I/usr/local/cuda/include -c assign_point_cuda.cu -o assign_point_cuda.o

# Compile o kernel OpenCL
gcc -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/include -I/usr/local/cuda/include -c assign_point_opencl.c -o assign_point_opencl.o

# Linke tudo
g++ -O3 -DSTARPU_USE_CUDA -DSTARPU_USE_OPENCL \
    kmeans_starpu.cpp assign_point_cuda.o assign_point_opencl.o -o kmeans_starpu_cuda_opencl \
    -I./starpu/include -I/usr/local/cuda/include \
    -L./starpu/src/.libs -L/usr/local/cuda/lib64 \
    -lstarpu-1.4 -lcuda -lcudart -lOpenCL -lpthread -lm
```

---

### 4. CPU + CUDA + OpenCL + MPI (paralelo distribuído)

```bash
mpicxx -O3 -DSTARPU_USE_CUDA -DSTARPU_USE_OPENCL \
    kmeans_starpu_prime.cpp assign_point_opencl.o assign_point_cuda.o -o kmeans_starpu_prime \
    -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/cuda/include \
    -L/home/bridge/starpu_install/lib -L/usr/local/cuda/lib64 \
    -lstarpu-1.4 -lcuda -lcudart -lOpenCL -lpthread -lm
```

---

> **Observações:**
> - Ajuste os caminhos de include/lib conforme seu ambiente.
> - Se não quiser OpenCL, basta não compilar/linkar `assign_point_opencl.c` e não usar `-DSTARPU_USE_OPENCL`.
> - Se não quiser CUDA, basta não compilar/linkar `assign_point_cuda.cu` e não usar `-DSTARPU_USE_CUDA`.
> - Se aparecer erro de referência a OpenCL, remova todas as referências a `assign_point_to_cluster_opencl` e `get_opencl_kernel_calls` do código.

---

## Execução

Antes de rodar, garanta que as bibliotecas estejam no `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/home/bridge/starpu_install/lib:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Execução Serial

```bash
./kmeans_starpu_cuda_opencl <INPUT> <K> <OUT-DIR>
```

### Execução Paralela com MPI

```bash
mpirun -np <NPROC> ./kmeans_starpu_prime <INPUT> <K> <OUT-DIR>
```
- `<NPROC>`: número de processos MPI (ex: 2, 4, 8...)

---

## Dicas de Troubleshooting

- **Verifique se a GPU está disponível:**  
  ```bash
  nvidia-smi
  ```
- **Verifique se o StarPU detecta CUDA:**  
  ```bash
  starpu_machine_display
  ```
  Procure por "CUDA worker" na saída.
- **Se não detectar GPU:**  
  - Confirme se o driver e toolkit CUDA estão instalados.
  - Confirme se o StarPU foi compilado com `--enable-cuda`.
  - Garanta que as bibliotecas CUDA estão no `LD_LIBRARY_PATH`.
  - Crie links simbólicos para `libcuda.so` em `/usr/local/cuda/lib64` se necessário.
- **Para OpenCL:**  
  - Confirme se o ICD e runtime estão instalados e visíveis.

---

Pronto! Agora seu ambiente está preparado para rodar em máquinas com CPU, GPU NVIDIA (CUDA), OpenCL (AMD/Intel/NVIDIA) e também em paralelo distribuído via MPI.


bridge-VJFH51F11X-B2211H% nvcc -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/include -I/usr/local/cuda/include -c assign_point_cuda.cu -o assign_point_cuda.o

# Compile o kernel OpenCL
gcc -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/include -I/usr/local/cuda/include -c assign_point_opencl.c -o assign_point_opencl.o

# Compile o binário principal com CUDA e OpenCL e MPI
mpicxx -O3 -DSTARPU_USE_CUDA -DSTARPU_USE_OPENCL \
    kmeans_starpu_prime.cpp assign_point_opencl.o assign_point_cuda.o -o kmeans_starpu_prime_cuda \
    -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/cuda/include \
    -L/home/bridge/starpu_install/lib -L/usr/local/cuda/lib64 \
    -lstarpu-1.4 -lcuda -lcudart -lOpenCL -lpthread -lm
zsh: command not found: #
zsh: command not found: #
kmeans_starpu_prime.cpp: In function ‘int main(int, char**)’:
kmeans_starpu_prime.cpp:517:16: warning: ignoring return value of ‘int starpu_init(starpu_conf*)’ declared with attribute ‘warn_unused_result’ [-Wunused-result]
  517 |     starpu_init(NULL);
      |     ~~~~~~~~~~~^~~~~~
kmeans_starpu_prime.cpp: In member function ‘void KMeans::assignPointsToClusters(std::vector<Point>&)’:
kmeans_starpu_prime.cpp:213:31: warning: ignoring return value of ‘int starpu_task_submit(starpu_task*)’ declared with attribute ‘warn_unused_result’ [-Wunused-result]
  213 |             starpu_task_submit(tasks[i]);
      |             ~~~~~~~~~~~~~~~~~~^~~~~~~~~~
bridge-VJFH51F11X-B2211H% 