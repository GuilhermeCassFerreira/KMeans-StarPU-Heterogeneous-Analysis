#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

/* ========================================================================== */
/* CONFIGURAÇÕES E UTILITÁRIOS                                                */
/* ========================================================================== */

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit((int)code);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

extern "C" {
volatile int cuda_assign_calls = 0;
volatile int cuda_calculate_calls = 0;
volatile int cuda_clean_calls = 0;
volatile int cuda_update_calls = 0;
volatile int cuda_accumulate_calls = 0;
static volatile int cuda_kernel_calls = 0;

/* ========================================================================== */
/* KERNELS DE NEGÓCIO (Assign)                                                */
/* ========================================================================== */

__global__ void assign_point_to_cluster_cuda_kernel(
    const double *points_values, const double *centroids,
    int K, int dimensions, int npoints, int *nearestClusterIds, int pts_per_block)
{
    // O lançamento limita as threads ativas por SM (ver assign_point_to_cluster_cuda),
    // então cada thread processa vários pontos. pts_per_block > 0: cada bloco fica com
    // uma faixa contínua de pts_per_block pontos; 0: grid-stride (lançamento antigo)
    int first = 0, last = npoints, step = gridDim.x * blockDim.x;
    if (pts_per_block > 0) {
        first = blockIdx.x * pts_per_block;
        last = min(first + pts_per_block, npoints);
        step = blockDim.x;
    } else {
        first = blockIdx.x * blockDim.x;
    }
    for (int idx = first + threadIdx.x; idx < last; idx += step) {
        const double *pv = points_values + (size_t)idx * dimensions;
        double min_dist = 1e300;
        int best = -1;

        // 4 centroides por vez: cada pv[d] é lido uma vez da memória global e
        // reaproveitado nos 4 (o acesso ao ponto não é coalescido, então reler
        // o ponto para cada centroide domina o tempo quando D é grande).
        // A ordem das somas e o desempate são os mesmos da versão de 1 em 1.
        int c = 0;
        for (; c + 3 < K; c += 4) {
            const double *c0 = centroids + c * dimensions;
            const double *c1 = c0 + dimensions;
            const double *c2 = c1 + dimensions;
            const double *c3 = c2 + dimensions;
            double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
            for (int d = 0; d < dimensions; ++d) {
                double p = pv[d];
                double e0 = p - c0[d], e1 = p - c1[d], e2 = p - c2[d], e3 = p - c3[d];
                d0 += e0 * e0; d1 += e1 * e1; d2 += e2 * e2; d3 += e3 * e3;
            }
            if (d0 < min_dist) { min_dist = d0; best = c; }
            if (d1 < min_dist) { min_dist = d1; best = c + 1; }
            if (d2 < min_dist) { min_dist = d2; best = c + 2; }
            if (d3 < min_dist) { min_dist = d3; best = c + 3; }
        }
        for (; c < K; ++c) {
            double dist = 0.0;
            const double *cent = centroids + c * dimensions;
            for (int d = 0; d < dimensions; ++d) {
                double diff = pv[d] - cent[d];
                dist += diff * diff;
            }
            if (dist < min_dist) { min_dist = dist; best = c; }
        }

        nearestClusterIds[idx] = best + 1;
    }
}

/* Distância expandida (KMEANS_ASSIGN_EXPANDIDO=1, só para teste): ‖x−c‖² = ‖x‖² − 2·x·c + ‖c‖².
 * ‖x‖² é o mesmo para todos os centroides e não muda a escolha, então basta comparar
 * ‖c‖² − 2·x·c: 1 FMA de FP64 por dimensão em vez de subtração + FMA. O kernel é limitado
 * pelo FP64, então o ganho vem direto daí. Cada bloco calcula os ‖c‖² na memória
 * compartilhada no início. O arredondamento muda: em quase empates um ponto pode ir para
 * outro centroide, então os centroides deixam de ser idênticos aos da versão normal. */
__global__ void assign_expandido_cuda_kernel(
    const double *points_values, const double *centroids,
    int K, int dimensions, int npoints, int *nearestClusterIds, int pts_per_block)
{
    extern __shared__ double s_norma[];   // ‖c‖² de cada centroide
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        const double *cent = centroids + (size_t)c * dimensions;
        double s = 0.0;
        for (int d = 0; d < dimensions; ++d) s += cent[d] * cent[d];
        s_norma[c] = s;
    }
    __syncthreads();

    int first = 0, last = npoints, step = gridDim.x * blockDim.x;
    if (pts_per_block > 0) {
        first = blockIdx.x * pts_per_block;
        last = min(first + pts_per_block, npoints);
        step = blockDim.x;
    } else {
        first = blockIdx.x * blockDim.x;
    }
    for (int idx = first + threadIdx.x; idx < last; idx += step) {
        const double *pv = points_values + (size_t)idx * dimensions;
        double min_dist = 1e300;
        int best = -1;
        int c = 0;
        for (; c + 3 < K; c += 4) {
            const double *c0 = centroids + c * dimensions;
            const double *c1 = c0 + dimensions;
            const double *c2 = c1 + dimensions;
            const double *c3 = c2 + dimensions;
            double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0;
            for (int d = 0; d < dimensions; ++d) {
                double p = pv[d];
                p0 += p * c0[d]; p1 += p * c1[d]; p2 += p * c2[d]; p3 += p * c3[d];
            }
            double d0 = s_norma[c] - 2.0 * p0, d1 = s_norma[c + 1] - 2.0 * p1;
            double d2 = s_norma[c + 2] - 2.0 * p2, d3 = s_norma[c + 3] - 2.0 * p3;
            if (d0 < min_dist) { min_dist = d0; best = c; }
            if (d1 < min_dist) { min_dist = d1; best = c + 1; }
            if (d2 < min_dist) { min_dist = d2; best = c + 2; }
            if (d3 < min_dist) { min_dist = d3; best = c + 3; }
        }
        for (; c < K; ++c) {
            const double *cent = centroids + c * dimensions;
            double prod = 0.0;
            for (int d = 0; d < dimensions; ++d) prod += pv[d] * cent[d];
            double dist = s_norma[c] - 2.0 * prod;
            if (dist < min_dist) { min_dist = dist; best = c; }
        }
        nearestClusterIds[idx] = best + 1;
    }
}

/* Vários threads por ponto (KMEANS_ASSIGN_THREADS_PONTO = 2 ou 4): cada um dos tpp threads
 * de um ponto calcula um subconjunto dos grupos de 4 centroides, e no fim eles trocam o melhor
 * resultado pelo warp. Cada distância é calculada com as mesmas operações, na mesma ordem, do
 * kernel normal, e o desempate continua indo para o menor índice, então os rótulos são
 * idênticos. Serve para chunks pequenos (SIFT com 64 chunks por nó): com um thread por ponto,
 * um chunk de 5 mil pontos deixa poucos threads por SM. */
__global__ void assign_multi_cuda_kernel(
    const double *points_values, const double *centroids,
    int K, int dimensions, int npoints, int *nearestClusterIds, int pts_per_block, int tpp)
{
    int sub = threadIdx.x % tpp;          // qual parte dos centroides este thread calcula
    int pts_passo = blockDim.x / tpp;     // pontos que o bloco processa por passada
    int first = blockIdx.x * pts_per_block;
    int last = min(first + pts_per_block, npoints);
    int ngrupos = K / 4;
    // base é igual em todo o bloco, então todos os threads de um warp chegam juntos ao shuffle
    for (int base = first; base < last; base += pts_passo) {
        int idx = base + threadIdx.x / tpp;
        double min_dist = 1e300;
        int best = INT_MAX;
        if (idx < last) {
            const double *pv = points_values + (size_t)idx * dimensions;
            for (int g = sub; g < ngrupos; g += tpp) {
                int c = 4 * g;
                const double *c0 = centroids + c * dimensions;
                const double *c1 = c0 + dimensions;
                const double *c2 = c1 + dimensions;
                const double *c3 = c2 + dimensions;
                double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
                for (int d = 0; d < dimensions; ++d) {
                    double p = pv[d];
                    double e0 = p - c0[d], e1 = p - c1[d], e2 = p - c2[d], e3 = p - c3[d];
                    d0 += e0 * e0; d1 += e1 * e1; d2 += e2 * e2; d3 += e3 * e3;
                }
                if (d0 < min_dist) { min_dist = d0; best = c; }
                if (d1 < min_dist) { min_dist = d1; best = c + 1; }
                if (d2 < min_dist) { min_dist = d2; best = c + 2; }
                if (d3 < min_dist) { min_dist = d3; best = c + 3; }
            }
            if (sub == 0) {   // centroides que sobram fora dos grupos de 4 (os de maior índice)
                for (int c = 4 * ngrupos; c < K; ++c) {
                    double dist = 0.0;
                    const double *cent = centroids + c * dimensions;
                    for (int d = 0; d < dimensions; ++d) {
                        double diff = pv[d] - cent[d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) { min_dist = dist; best = c; }
                }
            }
        }
        // Menor distância entre os tpp threads do ponto; no empate, o menor índice
        for (int off = tpp / 2; off > 0; off >>= 1) {
            double od = __shfl_xor_sync(0xffffffff, min_dist, off, tpp);
            int ob = __shfl_xor_sync(0xffffffff, best, off, tpp);
            if (od < min_dist || (od == min_dist && ob < best)) { min_dist = od; best = ob; }
        }
        if (idx < last && sub == 0) nearestClusterIds[idx] = best + 1;
    }
}

static int assign_threads_ponto() {
    static int v = 0;
    if (v == 0) {
        const char *e = getenv("KMEANS_ASSIGN_THREADS_PONTO");
        v = e ? atoi(e) : 1;
        if (v != 2 && v != 4) v = 1;
    }
    return v;
}

static bool assign_expandido() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KMEANS_ASSIGN_EXPANDIDO");
        v = e ? (atoi(e) == 1) : 0;
    }
    return v == 1;
}

/* KMEANS_ASSIGN_BALANCEADO=0 volta ao lançamento antigo (grid-stride), para comparar */
static bool assign_balanceado() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KMEANS_ASSIGN_BALANCEADO");
        v = e ? (atoi(e) == 1) : 1;
    }
    return v == 1;
}

/* KMEANS_ASSIGN_BLOCOS_SM: até quantos blocos de 128 threads por SM (padrão 2), para testar
 * se mais threads ativas compensam os pontos relidos saírem do cache */
static int assign_blocos_sm() {
    static int v = 0;
    if (v <= 0) {
        const char *e = getenv("KMEANS_ASSIGN_BLOCOS_SM");
        v = e ? atoi(e) : 2;
        if (v <= 0) v = 2;
    }
    return v;
}

/* Lança o Assign de um chunk na stream (usado pelo Assign e pelo Assign+Calculate) */
static void lancar_assign(cudaStream_t stream, const double *points_values, const double *centroids,
                          int *nearestClusterIds, int K, int dimensions, int chunk_size) {
    // Poucas threads ativas por SM: cada thread relê o próprio ponto (D doubles)
    // a cada grupo de 4 centroides, e com a GPU cheia (1536 threads/SM) esses
    // pontos não cabem no cache e as releituras vão para a DRAM. Com 256
    // threads/SM o kernel fica limitado pelo cálculo em FP64, não pela memória.
    static int cached_sm_count = 0;
    if (cached_sm_count == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached_sm_count, cudaDevAttrMultiProcessorCount, dev);
    }
    int max_blocks = cached_sm_count * assign_blocos_sm(); // padrão 2 blocos de 128 = 256 threads por SM
    int threads = 128, blocks, pts_per_block = 0;
    if (assign_balanceado()) {
        // Mesma quantidade de pontos por bloco, com até assign_blocos_sm() blocos por SM. Com o
        // grid-stride, um chunk pequeno deixava SMs com o dobro de pontos de outras
        // (SIFT na L4: 10 mil pontos em 79 blocos para 58 SMs); como o kernel é
        // limitado pelo FP64 de cada SM, ele terminava junto com a SM mais cheia
        pts_per_block = (chunk_size + max_blocks - 1) / max_blocks;
        threads = (pts_per_block + 31) / 32 * 32;
        if (threads > 128) threads = 128;
        blocks = (chunk_size + pts_per_block - 1) / pts_per_block;
    } else {
        blocks = (chunk_size + threads - 1) / threads;
        if (blocks > max_blocks) blocks = max_blocks;
    }

    int tpp = assign_expandido() ? 1 : assign_threads_ponto();
    if (tpp > 1) {
        // Mesma divisão balanceada de pontos por bloco, com tpp threads por ponto
        pts_per_block = (chunk_size + max_blocks - 1) / max_blocks;
        threads = (pts_per_block * tpp + 31) / 32 * 32;
        if (threads > 128) threads = 128;
        blocks = (chunk_size + pts_per_block - 1) / pts_per_block;
        assign_multi_cuda_kernel<<<blocks, threads, 0, stream>>>(
            points_values, centroids, K, dimensions, chunk_size,
            nearestClusterIds, pts_per_block, tpp);
    } else if (assign_expandido())
        assign_expandido_cuda_kernel<<<blocks, threads, K * sizeof(double), stream>>>(
            points_values, centroids, K, dimensions, chunk_size,
            nearestClusterIds, pts_per_block);
    else
        assign_point_to_cluster_cuda_kernel<<<blocks, threads, 0, stream>>>(
            points_values, centroids, K, dimensions, chunk_size,
            nearestClusterIds, pts_per_block);

    CUDA_CHECK(cudaGetLastError());
}

void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values     = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids         = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int    *nearestClusterIds = (int *)   STARPU_VECTOR_GET_PTR(buffers[2]);

    cuda_assign_calls++;
    cuda_kernel_calls++;

    lancar_assign(starpu_cuda_get_local_stream(), points_values, centroids, nearestClusterIds,
                  K, dimensions, chunk_size);
}

/* ========================================================================== */
/* KERNELS DE NEGÓCIO (Calculate Partial Sums)                                */
/* ========================================================================== */

__global__ void calculate_partial_sums_cuda_kernel(
    const double *points_values, const int *nearestClusterIds,
    int K, int dimensions, int npoints,
    double *partial_sums, int *partial_counts)
{
    extern __shared__ double shared_mem[];
    double *s_sums = shared_mem;
    int *s_counts = (int*)&s_sums[K * dimensions];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int total_vals = K * dimensions;

    for (int i = tid; i < total_vals; i += blockDim.x) {
        s_sums[i] = 0.0;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    if (idx < npoints) {
        int cluster_id = nearestClusterIds[idx] - 1;

        if (cluster_id >= 0 && cluster_id < K) {
            atomicAdd(&s_counts[cluster_id], 1);
            for (int d = 0; d < dimensions; ++d) {
                atomicAdd(&s_sums[cluster_id * dimensions + d], points_values[idx * dimensions + d]);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < total_vals; i += blockDim.x) {
        if (abs(s_sums[i]) > 1e-9) {
            atomicAdd(&partial_sums[i], s_sums[i]);
        }
    }
    for (int i = tid; i < K; i += blockDim.x) {
        if (s_counts[i] > 0) {
            atomicAdd(&partial_counts[i], s_counts[i]);
        }
    }
}

/* Versão sem memória compartilhada: usada quando as K*D somas não cabem no
 * bloco (ex.: K=256, D=128). Cada thread soma direto na memória global. */
__global__ void calculate_partial_sums_global_kernel(
    const double *points_values, const int *nearestClusterIds,
    int K, int dimensions, int npoints,
    double *partial_sums, int *partial_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    int cluster_id = nearestClusterIds[idx] - 1;
    if (cluster_id >= 0 && cluster_id < K) {
        atomicAdd(&partial_counts[cluster_id], 1);
        const double *pv = points_values + (size_t)idx * dimensions;
        double *ps = partial_sums + (size_t)cluster_id * dimensions;
        for (int d = 0; d < dimensions; ++d) {
            atomicAdd(&ps[d], pv[d]);
        }
    }
}

/* Lança o Calculate de um chunk na stream (usado pelo Calculate e pelo Assign+Calculate) */
static void lancar_calculate(cudaStream_t stream, const double *points_values, const int *nearestClusterIds,
                             double *partial_sums, int *partial_counts, int K, int dimensions, int npoints) {
    /* STARPU_W não zera o buffer automaticamente; zeramos antes da acumulação
     * assim como calculate_partial_sums_cpu faz com memset. */
    CUDA_CHECK(cudaMemsetAsync(partial_sums,   0, (size_t)K * dimensions * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(partial_counts, 0, (size_t)K * sizeof(int), stream));

    int threads = 256;
    int blocks = (npoints + threads - 1) / threads;

    size_t shared_mem_size = (K * dimensions * sizeof(double)) + (K * sizeof(int));

    static size_t cached_shared_mem_limit = 0;
    if (cached_shared_mem_limit == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        cached_shared_mem_limit = prop.sharedMemPerBlock;
    }

    // K*D somas não cabem na memória compartilhada do bloco: usa a versão em memória global
    if (shared_mem_size > cached_shared_mem_limit) {
        calculate_partial_sums_global_kernel<<<blocks, threads, 0, stream>>>(
            points_values, nearestClusterIds, K, dimensions, npoints, partial_sums, partial_counts);
    } else {
        calculate_partial_sums_cuda_kernel<<<blocks, threads, shared_mem_size, stream>>>(
            points_values, nearestClusterIds, K, dimensions, npoints, partial_sums, partial_counts);
    }

    CUDA_CHECK(cudaGetLastError());
}

void calculate_partial_sums_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    cuda_calculate_calls++;
    cuda_kernel_calls++;

    lancar_calculate(starpu_cuda_get_local_stream(), points_values, nearestClusterIds,
                     partial_sums, partial_counts, K, dimensions, chunk_size);
}

/* Assign + Calculate do chunk na mesma tarefa: os dois kernels em sequência na stream do
 * worker, com os rótulos e a soma do chunk ficando na GPU que fez o Assign */
void assign_calc_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values     = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids         = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int    *nearestClusterIds = (int *)   STARPU_VECTOR_GET_PTR(buffers[2]);
    double *partial_sums      = (double *)STARPU_VECTOR_GET_PTR(buffers[3]);
    int    *partial_counts    = (int *)   STARPU_VECTOR_GET_PTR(buffers[4]);

    cuda_assign_calls++;
    cuda_calculate_calls++;
    cuda_kernel_calls++;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    lancar_assign(stream, points_values, centroids, nearestClusterIds, K, dimensions, chunk_size);
    lancar_calculate(stream, points_values, nearestClusterIds, partial_sums, partial_counts,
                     K, dimensions, chunk_size);
}

/* ========================================================================== */
/* KERNELS DE LIMPEZA E ATUALIZAÇÃO (CUDA)                                    */
/* ========================================================================== */

__global__ void clean_buffers_cuda_kernel(double *partial_sums, int *partial_counts, int total_doubles, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_doubles) {
        partial_sums[idx] = 0.0;
    }
    if (idx < K) {
        partial_counts[idx] = 0;
    }
}

void clean_buffers_cuda(void *buffers[], void *cl_arg) {
    cuda_clean_calls++;
    cuda_kernel_calls++;

    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums   = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int    *partial_counts = (int *)   STARPU_VECTOR_GET_PTR(buffers[1]);

    int total_doubles = K * dimensions;
    int threads = 256;
    int blocks = (total_doubles + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    clean_buffers_cuda_kernel<<<blocks, threads, 0, stream>>>(partial_sums, partial_counts, total_doubles, K);
}


__global__ void update_centroids_cuda_kernel(double *partial_sums, int *partial_counts,
                                             double *centroids, int K, int dimensions) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K && partial_counts[c] > 0) {
        for (int d = 0; d < dimensions; ++d) {
            centroids[c * dimensions + d] =
                partial_sums[c * dimensions + d] / partial_counts[c];
        }
    }
}

void update_centroids_cuda(void *buffers[], void *cl_arg) {
    cuda_update_calls++;
    cuda_kernel_calls++;

    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums   = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int    *partial_counts = (int *)   STARPU_VECTOR_GET_PTR(buffers[1]);
    double *centroids      = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int threads = 256;
    int blocks  = (K + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    update_centroids_cuda_kernel<<<blocks, threads, 0, stream>>>(
        partial_sums, partial_counts, centroids, K, dimensions);
}

__global__ void accumulate_nodes_cuda_kernel(double *master_sums, int *master_counts, 
                                           double *node_sums, int *node_counts, 
                                           int K, int dimensions) {
    // Uma thread por elemento das K*D somas: threads vizinhas acessam endereços
    // vizinhos (coalescido). Cada elemento recebe a mesma soma de antes.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K * dimensions) master_sums[i] += node_sums[i];
    if (i < K) master_counts[i] += node_counts[i];
}

extern "C" void accumulate_nodes_cuda(void *buffers[], void *cl_arg) {
    cuda_accumulate_calls++;
    cuda_kernel_calls++;

    int K, dimensions;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions);

    double *master_sums   = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int    *master_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *node_sums     = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int    *node_counts   = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int threads = 256;
    int blocks = (K * dimensions + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    accumulate_nodes_cuda_kernel<<<blocks, threads, 0, stream>>>(
        master_sums, master_counts, node_sums, node_counts, K, dimensions
    );
    CUDA_CHECK(cudaGetLastError());
}


int get_cuda_kernel_calls() { return cuda_kernel_calls; }

} // extern "C"
