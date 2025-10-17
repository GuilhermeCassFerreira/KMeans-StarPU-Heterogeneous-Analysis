#include <cuda_runtime.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <cstdio>
#include <cstdlib>

struct HandleArgs {
    int K;
    int dimensions;
    int chunk_size; // número de pontos no chunk
};

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit((int)code);
    }
}

extern "C" {

// Contadores exportados para serem lidos pelo código C++
int cuda_assign_calls = 0;
int cuda_calculate_calls = 0;
static int cuda_kernel_calls = 0; // total de chamadas de kernels CUDA (agregado)

/* Implementação de atomicAdd para double (caso não disponível) */
__device__ double atomicAddDouble(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/* Kernel CUDA: cada thread processa um ponto do chunk (distância ao quadrado). */
__global__ void assign_point_to_cluster_cuda_kernel(
    const double *points_values, const double *centroids, int K, int dimensions, int npoints, int *nearestClusterIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    const double *pv = points_values + (size_t)idx * dimensions;
    double min_dist = 1e300;
    int best = -1;

    for (int c = 0; c < K; ++c) {
        double dist = 0.0;
        for (int d = 0; d < dimensions; ++d) {
            double diff = pv[d] - centroids[c * dimensions + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }

    nearestClusterIds[idx] = best + 1;
}

/* Wrapper chamado pelo StarPU quando a tarefa for executada na GPU. */
void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {

    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) return;

    // incrementa contadores visíveis externamente
    ++cuda_assign_calls;
    ++cuda_kernel_calls;

    static bool assign_cuda_printed = false;
    if (!assign_cuda_printed) {
        printf("[KERNEL] assign_point_to_cluster CUDA executed\n");
        fflush(stdout);
        assign_cuda_printed = true;
    }

    double *points_values     = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids         = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int    *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int K = args->K;
    int dimensions = args->dimensions;
    int npoints = args->chunk_size; // número correto de pontos no chunk

    if (!points_values || !centroids || !nearestClusterIds || npoints <= 0) return;

    int threads = 256;
    int blocks = (npoints + threads - 1) / threads;

    cudaStream_t stream = starpu_cuda_get_local_stream();
    assign_point_to_cluster_cuda_kernel<<<blocks, threads, 0, stream>>>(
        points_values, centroids, K, dimensions, npoints, nearestClusterIds);

    CUDA_CHECK(cudaGetLastError());
    // garantir que a stream local do StarPU terminou antes de retornar
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Kernel CUDA: cada thread processa um ponto do chunk e acumula somas parciais */
__global__ void calculate_partial_sums_cuda_kernel(
    const double *points_values, const int *nearestClusterIds, int K, int dimensions, int npoints,
    double *partial_sums, int *partial_counts)
{
    extern __shared__ double shared_sums[]; // Memória compartilhada para somas parciais
    int *shared_counts = (int *)&shared_sums[K * dimensions];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    for (int k = tid; k < K * dimensions; k += blockDim.x) {
        shared_sums[k] = 0.0;
    }
    for (int k = tid; k < K; k += blockDim.x) {
        shared_counts[k] = 0;
    }
    __syncthreads();

    if (idx < npoints) {
        int cluster_id = nearestClusterIds[idx] - 1;
        if (cluster_id >= 0 && cluster_id < K) {
            for (int d = 0; d < dimensions; ++d) {
                atomicAddDouble(&shared_sums[cluster_id * dimensions + d], points_values[idx * dimensions + d]);
            }
            atomicAdd(&shared_counts[cluster_id], 1);
        }
    }
    __syncthreads();

    for (int k = tid; k < K * dimensions; k += blockDim.x) {
        atomicAddDouble(&partial_sums[k], shared_sums[k]);
    }
    for (int k = tid; k < K; k += blockDim.x) {
        atomicAdd(&partial_counts[k], shared_counts[k]);
    }
}

/* Wrapper chamado pelo StarPU para calcular somas parciais na GPU */
void calculate_partial_sums_cuda(void *buffers[], void *cl_arg) {
    // incrementa contadores visíveis externamente
    ++cuda_calculate_calls;
    ++cuda_kernel_calls;

    static bool calculate_cuda_printed = false;
    if (!calculate_cuda_printed) {
        printf("[KERNEL] calculate_partial_sums CUDA executed\n");
        fflush(stdout);
        calculate_cuda_printed = true;
    }

    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) return;

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int K = args->K;
    int dimensions = args->dimensions;
    int npoints = args->chunk_size; // número correto de pontos no chunk

    if (!points_values || !nearestClusterIds || !partial_sums || !partial_counts || npoints <= 0) return;

    int threads = 256;
    int blocks = (npoints + threads - 1) / threads;

    size_t shared_mem_size = K * dimensions * sizeof(double) + K * sizeof(int);

    cudaStream_t stream = starpu_cuda_get_local_stream();
    calculate_partial_sums_cuda_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        points_values, nearestClusterIds, K, dimensions, npoints, partial_sums, partial_counts);

    CUDA_CHECK(cudaGetLastError());
    // garantir que a stream local do StarPU terminou antes de retornar
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
}

int get_cuda_kernel_calls() { return cuda_kernel_calls; }

}