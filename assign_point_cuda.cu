#include <cuda_runtime.h>
#include <starpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

struct HandleArgs {
    int K;
    int dimensions;
    int chunk_size;
};

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{ if (code != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line); if (abort) exit((int)code); } }

extern "C" {

static int cuda_kernel_calls = 0;

/* Kernel CUDA: cada thread processa um ponto do chunk.
   Usa distâncias ao quadrado (sem sqrt) para menor custo. */
__global__ void assign_point_to_cluster_cuda_kernel(
    const double *points_values, const double *centroids, int K, int dimensions, int chunk_size, int *nearestClusterIds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;

    const double *pv = points_values + (size_t)idx * dimensions;
    double min_dist = 1e300;
    int best = -1;

    for (int c = 0; c < K; ++c) {
        const double *cent = centroids + (size_t)c * dimensions;
        double dist = 0.0;
        for (int d = 0; d < dimensions; ++d) {
            double diff = cent[d] - pv[d];
            dist += diff * diff;
        }
        if (dist < min_dist) { min_dist = dist; best = c; }
    }

    nearestClusterIds[idx] = best + 1;
}

/* Wrapper chamado pelo StarPU quando a tarefa for executada na GPU.
   Assume que STARPU fornece ponteiros corretos (dispositivo) via STARPU_VECTOR_GET_PTR. */
void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {

    cuda_kernel_calls++;

    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) return;

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int K = args->K;
    int dimensions = args->dimensions;
    int chunk_size = args->chunk_size;
    if (!points_values || !centroids || !nearestClusterIds || chunk_size <= 0) return;

    int threads = 256;
    int blocks = (chunk_size + threads - 1) / threads;

    assign_point_to_cluster_cuda_kernel<<<blocks, threads>>>(
        points_values, centroids, K, dimensions, chunk_size, nearestClusterIds);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int get_cuda_kernel_calls() { return cuda_kernel_calls; }

} //