#include <cuda_runtime.h>
#include <starpu.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

struct StarPUArgs {
    double *points_values;      // [chunk_size][dimensions] flattened
    int *nearestClusterIds;     // pointer to chunk output
    double *centroids;          // [K][dimensions] flattened
    int K;
    int dimensions;
    int offset;
    int chunk_size;
};

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{ if (code != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line); if (abort) exit((int)code); } }

extern "C" {

static int cuda_kernel_calls = 0;

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
        dist = sqrt(dist);
        if (dist < min_dist) { min_dist = dist; best = c; }
    }
    nearestClusterIds[idx] = best + 1;
}

void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {
    StarPUArgs *args = (StarPUArgs *)cl_arg;
    if (!args) return;

    int chunk_size = args->chunk_size;
    int dimensions = args->dimensions;
    int K = args->K;

    if (!args->points_values || !args->centroids || !args->nearestClusterIds || chunk_size <= 0 || dimensions <= 0 || K <= 0) {
        fprintf(stderr, "[CUDA] invalid args\n"); return;
    }

    size_t points_bytes = (size_t)chunk_size * dimensions * sizeof(double);
    size_t cent_bytes = (size_t)K * dimensions * sizeof(double);
    size_t out_bytes = (size_t)chunk_size * sizeof(int);

    double *d_points = nullptr;
    double *d_cent = nullptr;
    int *d_out = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_points, points_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_cent, cent_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_points, args->points_values, points_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent, args->centroids, cent_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (chunk_size + threads - 1) / threads;
    assign_point_to_cluster_cuda_kernel<<<blocks, threads>>>(d_points, d_cent, K, dimensions, chunk_size, d_out);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(args->nearestClusterIds, d_out, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_points);
    cudaFree(d_cent);
    cudaFree(d_out);

    ++cuda_kernel_calls;
}

int get_cuda_kernel_calls() { return cuda_kernel_calls; }

} // extern "C"