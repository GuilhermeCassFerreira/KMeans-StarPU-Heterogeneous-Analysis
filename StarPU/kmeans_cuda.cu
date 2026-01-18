#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* -------------------------------------------------------------------------- */
/* CONFIGURAÇÕES E UTILITÁRIOS                                                */
/* -------------------------------------------------------------------------- */

struct HandleArgs {
    int K;
    int dimensions;
    int chunk_size;
};

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit((int)code);
    }
}

/* Implementação de atomicAdd para double em arquiteturas Pascal ou inferiores (< sm_60) */
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

// Contadores para métricas
int cuda_assign_calls = 0;
int cuda_calculate_calls = 0;
static int cuda_kernel_calls = 0;

/* -------------------------------------------------------------------------- */
/* KERNELS DE NEGÓCIO (Assign & Calculate)                                    */
/* -------------------------------------------------------------------------- */

/*
 * Kernel: assign_point_to_cluster
 * Determina qual o centroide mais próximo para cada ponto.
 */
__global__ void assign_point_to_cluster_cuda_kernel(
    const double *points_values, const double *centroids, int K, int dimensions, int npoints, int *nearestClusterIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    // Ponteiro para o ponto atual
    const double *pv = points_values + (size_t)idx * dimensions;
    double min_dist = 1e300;
    int best = -1;

    for (int c = 0; c < K; ++c) {
        double dist = 0.0;
        const double *cent = centroids + c * dimensions;
        
        // Loop desenrolado ou simples para cálculo da distância euclidiana
        for (int d = 0; d < dimensions; ++d) {
            double diff = pv[d] - cent[d];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }

    nearestClusterIds[idx] = best + 1; // +1 porque 0 pode ser usado como 'nulo' em alguns contextos, ou manter consistência
}

/*
 * Wrapper StarPU: assign_point_to_cluster_cuda
 * Buffers: [0] Points(R), [1] Centroids(R), [2] Labels(W)
 */
void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {
    HandleArgs *args = (HandleArgs *)cl_arg;
    
    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int K = args->K;
    int dimensions = args->dimensions;
    int npoints = args->chunk_size;

    cuda_assign_calls++;
    cuda_kernel_calls++;

    int threads = 256;
    int blocks = (npoints + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    assign_point_to_cluster_cuda_kernel<<<blocks, threads, 0, stream>>>(
        points_values, centroids, K, dimensions, npoints, nearestClusterIds);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/*
 * Kernel: calculate_partial_sums
 * Acumula as somas e contagens. Otimizado com Shared Memory para reduzir atômicos globais.
 */
__global__ void calculate_partial_sums_cuda_kernel(
    const double *points_values, const int *nearestClusterIds, int K, int dimensions, int npoints,
    double *partial_sums, int *partial_counts)
{
    // Memória compartilhada dinâmica
    // Layout: [Sums (K*dim doubles)] [Counts (K ints)]
    extern __shared__ double shared_mem[];
    double *s_sums = shared_mem;
    int *s_counts = (int*)&s_sums[K * dimensions];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int total_vals = K * dimensions;

    // 1. Inicializar Shared Memory com 0
    for (int i = tid; i < total_vals; i += blockDim.x) {
        s_sums[i] = 0.0;
    }
    for (int i = tid; i < K; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    // 2. Acumular na Shared Memory
    if (idx < npoints) {
        int cluster_id = nearestClusterIds[idx] - 1; // Ajuste se for base 1
        
        // Verificação de segurança
        if (cluster_id >= 0 && cluster_id < K) {
            // Atômico na Shared Memory (muito rápido)
            atomicAdd(&s_counts[cluster_id], 1);
            
            for (int d = 0; d < dimensions; ++d) {
                atomicAdd(&s_sums[cluster_id * dimensions + d], points_values[idx * dimensions + d]);
            }
        }
    }
    __syncthreads();

    // 3. Descarregar Shared Memory para Global Memory (Buffer de Redução do StarPU)
    // Usamos atômicos globais aqui, mas apenas uma vez por bloco, não por ponto.
    for (int i = tid; i < total_vals; i += blockDim.x) {
        if (abs(s_sums[i]) > 1e-9) { // Pequena otimização: não somar zeros
            atomicAdd(&partial_sums[i], s_sums[i]);
        }
    }
    for (int i = tid; i < K; i += blockDim.x) {
        if (s_counts[i] > 0) {
            atomicAdd(&partial_counts[i], s_counts[i]);
        }
    }
}

/*
 * Wrapper StarPU: calculate_partial_sums_cuda
 * Buffers: [0] Points(R), [1] Labels(R), [2] Sums(REDUX), [3] Counts(REDUX)
 */
void calculate_partial_sums_cuda(void *buffers[], void *cl_arg) {
    HandleArgs *args = (HandleArgs *)cl_arg;
    
    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    
    // Estes ponteiros apontam para buffers temporários ou persistentes na GPU gerenciados pelo Redux
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int K = args->K;
    int dimensions = args->dimensions;
    int npoints = args->chunk_size;

    cuda_calculate_calls++;
    cuda_kernel_calls++;

    int threads = 256;
    int blocks = (npoints + threads - 1) / threads;

    // Calcular tamanho da Shared Memory necessária
    size_t shared_mem_size = (K * dimensions * sizeof(double)) + (K * sizeof(int));
    
    // Obter propriedades do device para verificar limite de shared memory
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    
    // Se a memória compartilhada necessária for maior que a disponível, 
    // teríamos que usar uma versão sem shared memory (apenas atômicos globais).
    // Aqui assumimos que cabe (K razoável). Se não couber, o kernel falhará na execução.
    // Em produção, adicione um fallback.
    if (shared_mem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "[KMeans CUDA] WARN: Shared memory insuficiente. K muito grande? Falha provável.\n");
    }

    cudaStream_t stream = starpu_cuda_get_local_stream();

    calculate_partial_sums_cuda_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        points_values, nearestClusterIds, K, dimensions, npoints, partial_sums, partial_counts);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/* -------------------------------------------------------------------------- */
/* KERNELS DE REDUÇÃO (REDUX)                                                 */
/* Necessários para o StarPU combinar resultados parciais                     */
/* -------------------------------------------------------------------------- */

// --- DOUBLE (Somas) ---

// Init: Zera o buffer
__global__ void redux_double_init_kernel(double *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = 0.0;
    }
}

// Reduce: Dst += Src
__global__ void redux_double_reduce_kernel(double *dst, const double *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] += src[i];
    }
}

void redux_double_init_cuda(void *buffers[], void *cl_arg) {
    double *dst = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]); // Número de elementos
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Cap grid size

    redux_double_init_kernel<<<blocks, threads, 0, stream>>>(dst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void redux_double_reduce_cuda(void *buffers[], void *cl_arg) {
    double *dst = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *src = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    redux_double_reduce_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// --- INT (Contagens) ---

// Init: Zera o buffer
__global__ void redux_int_init_kernel(int *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = 0;
    }
}

// Reduce: Dst += Src
__global__ void redux_int_reduce_kernel(int *dst, const int *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] += src[i];
    }
}

void redux_int_init_cuda(void *buffers[], void *cl_arg) {
    int *dst = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    redux_int_init_kernel<<<blocks, threads, 0, stream>>>(dst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void redux_int_reduce_cuda(void *buffers[], void *cl_arg) {
    int *dst = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *src = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    redux_int_reduce_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Função auxiliar para retornar chamadas ao main
int get_cuda_kernel_calls() { return cuda_kernel_calls; }

} // extern "C"