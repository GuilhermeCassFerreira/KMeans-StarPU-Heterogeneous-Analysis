#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

/* Ponteiro para o flag de convergência na CPU (pinned memory).
 * Definido via starpu_set_converged_cpu_ptr() antes da submissão das tarefas.
 * Usado em update_centroids_cuda para copiar *converged GPU→CPU na mesma stream,
 * garantindo que o callback do StarPU veja o valor correto ao disparar. */
static int *g_converged_cpu_ptr = nullptr;

extern "C" void starpu_set_converged_cpu_ptr(int *ptr) {
    g_converged_cpu_ptr = ptr;
}

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
    int K, int dimensions, int npoints, int *nearestClusterIds,
    int *converged, int *local_changes)
{
    if (*converged == 1) return; // ghost task: iteração já convergiu

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    const double *pv = points_values + (size_t)idx * dimensions;
    double min_dist = 1e300;
    int best = -1;

    for (int c = 0; c < K; ++c) {
        double dist = 0.0;
        const double *cent = centroids + c * dimensions;
        for (int d = 0; d < dimensions; ++d) {
            double diff = pv[d] - cent[d];
            dist += diff * diff;
        }
        if (dist < min_dist) { min_dist = dist; best = c; }
    }

    int new_label = best + 1;
    if (nearestClusterIds[idx] != new_label) {
        nearestClusterIds[idx] = new_label;
        atomicAdd(local_changes, 1); // REDUX: acumula na cópia privada deste chunk
    }
}

void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values     = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids         = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int    *nearestClusterIds = (int *)   STARPU_VECTOR_GET_PTR(buffers[2]);
    int    *converged         = (int *)   STARPU_VARIABLE_GET_PTR(buffers[3]);
    int    *local_changes     = (int *)   STARPU_VARIABLE_GET_PTR(buffers[4]); // REDUX

    cuda_assign_calls++;
    cuda_kernel_calls++;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    int threads = 256;
    int blocks  = (chunk_size + threads - 1) / threads;

    assign_point_to_cluster_cuda_kernel<<<blocks, threads, 0, stream>>>(
        points_values, centroids, K, dimensions, chunk_size,
        nearestClusterIds, converged, local_changes);

    CUDA_CHECK(cudaGetLastError());
}

/* ========================================================================== */
/* KERNELS DE NEGÓCIO (Calculate Partial Sums)                                */
/* ========================================================================== */

__global__ void calculate_partial_sums_cuda_kernel(
    const double *points_values, const int *nearestClusterIds,
    int K, int dimensions, int npoints,
    double *partial_sums, int *partial_counts, int *converged)
{
    if (*converged == 1) return; 

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

void calculate_partial_sums_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);
    int *converged = (int *)STARPU_VARIABLE_GET_PTR(buffers[4]);

    int npoints = chunk_size;

    cuda_calculate_calls++;
    cuda_kernel_calls++;

    cudaStream_t stream = starpu_cuda_get_local_stream();

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

    if (shared_mem_size > cached_shared_mem_limit) {
        fprintf(stderr, "[KMeans CUDA] WARN: Shared memory insuficiente. K muito grande? Falha provavel.\n");
    }

    calculate_partial_sums_cuda_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        points_values, nearestClusterIds, K, dimensions, npoints, partial_sums, partial_counts, converged);

    CUDA_CHECK(cudaGetLastError());
}

/* ========================================================================== */
/* KERNELS DE LIMPEZA E ATUALIZAÇÃO (CUDA)                                    */
/* ========================================================================== */

__global__ void clean_buffers_cuda_kernel(double *partial_sums, int *partial_counts, int total_doubles, int K, int *converged) {
    if (*converged == 1) return; 

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
    int    *converged      = (int *)   STARPU_VARIABLE_GET_PTR(buffers[2]);

    int total_doubles = K * dimensions;
    int threads = 256;
    int blocks = (total_doubles + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    clean_buffers_cuda_kernel<<<blocks, threads, 0, stream>>>(partial_sums, partial_counts, total_doubles, K, converged);
}


__global__ void update_centroids_cuda_kernel(double *partial_sums, int *partial_counts,
                                             double *centroids, int K, int dimensions,
                                             int *converged, int *total_changes) {
    if (*converged == 1) return;

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K && partial_counts[c] > 0) {
        for (int d = 0; d < dimensions; ++d) {
            centroids[c * dimensions + d] =
                partial_sums[c * dimensions + d] / partial_counts[c];
        }
    }
    // Mesmo critério do OMP/CPU: zero mudanças de label → convergência
    if (c == 0 && *total_changes == 0) {
        *converged = 1;
    }
    if (c == 0) {
        *total_changes = 0; // reseta para a próxima iteração (REDUX parte de 0)
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
    int    *converged      = (int *)   STARPU_VARIABLE_GET_PTR(buffers[3]);
    int    *total_changes  = (int *)   STARPU_VARIABLE_GET_PTR(buffers[4]); // REDUX result

    int threads = 256;
    int blocks  = (K + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    update_centroids_cuda_kernel<<<blocks, threads, 0, stream>>>(
        partial_sums, partial_counts, centroids, K, dimensions, converged, total_changes);

    /* Copia o flag de convergência GPU→CPU na mesma stream, ANTES do evento de
     * conclusão do StarPU. Assim o callback vê o valor correto em CPU memory. */
    if (g_converged_cpu_ptr)
        CUDA_CHECK(cudaMemcpyAsync(g_converged_cpu_ptr, converged, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
}

__global__ void accumulate_nodes_cuda_kernel(double *master_sums, int *master_counts, 
                                           double *node_sums, int *node_counts, 
                                           int K, int dimensions, int *converged) {
    if (*converged == 1) return; 

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        master_counts[c] += node_counts[c];
        for (int d = 0; d < dimensions; d++) {
            master_sums[c * dimensions + d] += node_sums[c * dimensions + d];
        }
    }
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
    int    *converged     = (int *)STARPU_VARIABLE_GET_PTR(buffers[4]); 

    int threads = 256;
    int blocks = (K + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    accumulate_nodes_cuda_kernel<<<blocks, threads, 0, stream>>>(
        master_sums, master_counts, node_sums, node_counts, K, dimensions, converged
    );
}


int get_cuda_kernel_calls() { return cuda_kernel_calls; }

/* REDUX para h_changes: init zera a cópia privada na GPU. */
extern "C" void changes_var_init_cuda(void *buffers[], void *) {
    int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    CUDA_CHECK(cudaMemsetAsync(val, 0, sizeof(int), starpu_cuda_get_local_stream()));
}

/* REDUX reduce na GPU: dst += src (para o caso NCPU=0 onde só há workers CUDA). */
__global__ void reduce_int_var_kernel(int *dst, const int *src) {
    *dst += *src;
}

extern "C" void changes_var_reduce_cuda(void *buffers[], void *) {
    int *dst = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const int *src = (const int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    reduce_int_var_kernel<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(dst, src);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"