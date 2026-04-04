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

extern "C" {

/* ========================================================================== */
/* Contadores para métricas                                                   */
/* ========================================================================== */

int cuda_assign_calls = 0;
int cuda_calculate_calls = 0;
static int cuda_kernel_calls = 0;

/* ========================================================================== */
/* KERNELS DE NEGÓCIO (Assign)                                                */
/* ========================================================================== */

__global__ void assign_point_to_cluster_cuda_kernel(
    const double *points_values, const double *centroids,
    int K, int dimensions, int npoints, int *nearestClusterIds)
{
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

        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }

    nearestClusterIds[idx] = best + 1;
}

void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int npoints = chunk_size;

    cuda_assign_calls++;
    cuda_kernel_calls++;

    int threads = 256;
    int blocks = (npoints + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();

    assign_point_to_cluster_cuda_kernel<<<blocks, threads, 0, stream>>>(
        points_values, centroids, K, dimensions, npoints, nearestClusterIds);

    CUDA_CHECK(cudaGetLastError());
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

void calculate_partial_sums_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int npoints = chunk_size;

    cuda_calculate_calls++;
    cuda_kernel_calls++;

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

    cudaStream_t stream = starpu_cuda_get_local_stream();

    calculate_partial_sums_cuda_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        points_values, nearestClusterIds, K, dimensions, npoints, partial_sums, partial_counts);

    CUDA_CHECK(cudaGetLastError());
}

/* ========================================================================== */
/* KERNELS DE REDUÇÃO (REDUX)                                                 */
/* ========================================================================== */

__global__ void redux_double_init_kernel(double *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = 0.0;
    }
}

__global__ void redux_double_reduce_kernel(double *dst, const double *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] += src[i];
    }
}

void redux_double_init_cuda(void *buffers[], void *cl_arg) {
    double *dst = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

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

__global__ void redux_int_init_kernel(int *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = 0;
    }
}

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
    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);

    int total_doubles = K * dimensions;
    int threads = 256;
    int blocks = (total_doubles + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    clean_buffers_cuda_kernel<<<blocks, threads, 0, stream>>>(partial_sums, partial_counts, total_doubles, K);
    cudaStreamSynchronize(stream);
}

__global__ void update_centroids_cuda_kernel(double *partial_sums, int *partial_counts, double *centroids, int K, int dimensions) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        if (partial_counts[c] > 0) {
            for (int d = 0; d < dimensions; ++d) {
                centroids[c * dimensions + d] = partial_sums[c * dimensions + d] / partial_counts[c];
            }
        }
    }
}

void update_centroids_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int threads = 256;
    int blocks = (K + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    update_centroids_cuda_kernel<<<blocks, threads, 0, stream>>>(partial_sums, partial_counts, centroids, K, dimensions);
    cudaStreamSynchronize(stream);
}

__global__ void accumulate_nodes_cuda_kernel(double *master_sums, int *master_counts, 
                                           double *node_sums, int *node_counts, 
                                           int K, int dimensions) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        // Soma as contagens de pontos
        master_counts[c] += node_counts[c];
        
        // Soma as coordenadas acumuladas
        for (int d = 0; d < dimensions; d++) {
            master_sums[c * dimensions + d] += node_sums[c * dimensions + d];
        }
    }
}

extern "C" void accumulate_nodes_cuda(void *buffers[], void *cl_arg) {
    int K, dimensions;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions);

    double *master_sums   = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int    *master_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *node_sums     = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int    *node_counts   = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int threads = 256;
    int blocks = (K + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    accumulate_nodes_cuda_kernel<<<blocks, threads, 0, stream>>>(
        master_sums, master_counts, node_sums, node_counts, K, dimensions
    );
}

int get_cuda_kernel_calls() { return cuda_kernel_calls; }

} // extern "C"