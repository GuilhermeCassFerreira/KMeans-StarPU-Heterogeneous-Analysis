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

- **Para MPI (opcional):**
  - Instale o OpenMPI ou MPICH:
    ```bash
    sudo apt-get install -y libopenmpi-dev openmpi-bin
    ```

---

### 2. Instalando Dependências no Host

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ocl-icd-opencl-dev libhwloc-dev wget pkg-config vim
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

---

### 3. Instalando a StarPU (se necessário)

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

### 4. Configurando Variáveis de Ambiente

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export STARPU_NCUDA=1
```

---

## Compilação

### 1. Compilando a versão **StarPU** (heterogênea)

```bash
g++ -O3 -DSTARPU_USE_CUDA -DSTARPU_USE_OPENCL \
    kmeans_starpu_single.cpp core_affinity.cpp assign_point_cuda.o assign_point_opencl.o -o kmeans_starpu_cuda_opencl \
    -I/home/bridge/starpu_install/include/starpu/1.4 -I/usr/local/cuda/include \
    -L/home/bridge/starpu_install/lib -L/usr/local/cuda/lib64 \
    -lstarpu-1.4 -lcuda -lcudart -lOpenCL -lpthread -lm
```

---

### 2. Compilando a versão **sequencial**

```bash
g++ -O3 kmeans.cpp -o kmeans_sequencial \
    -lpthread -lm
```

---

### 3. Compilando o gerador de entradas

O arquivo `generate_input.py` não precisa de compilação, pois é um script Python. Certifique-se de que o Python está instalado no sistema.

---

## Execução

### 1. Gerando o arquivo de entrada

Use o script `generate_input.py` para gerar um arquivo de entrada massivo:

```bash
python3 generate_input.py
```

Por padrão, o script gera um arquivo chamado `massive_input.txt` com 50 milhões de pontos em 2 dimensões. Você pode ajustar os parâmetros no script.

---

### 2. Executando a versão **StarPU**

```bash
./kmeans_starpu_cuda_opencl <INPUT> <K> <OUT-DIR> [CHUNK_SIZE]
```

- `<INPUT>`: Caminho para o arquivo de entrada (ex.: `massive_input.txt`).
- `<K>`: Número de clusters.
- `<OUT-DIR>`: Diretório onde os resultados serão salvos.
- `[CHUNK_SIZE]`: (Opcional) Tamanho dos chunks. Padrão: 100.

#### **Exemplo:**
```bash
./kmeans_starpu_cuda_opencl massive_input.txt 4 ./output 128
```

---

### 3. Executando a versão **sequencial**

```bash
./kmeans_sequencial <INPUT> <K> <OUT-DIR>
```

- `<INPUT>`: Caminho para o arquivo de entrada (ex.: `massive_input.txt`).
- `<K>`: Número de clusters.
- `<OUT-DIR>`: Diretório onde os resultados serão salvos.

#### **Exemplo:**
```bash
./kmeans_sequencial massive_input.txt 4 ./output
```

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

Pronto! Agora você pode compilar e executar as versões do K-Means com StarPU (heterogênea) e sequencial, além de gerar entradas massivas para os testes.
