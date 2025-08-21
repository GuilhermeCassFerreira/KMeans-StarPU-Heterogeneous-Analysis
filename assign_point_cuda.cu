#include <cuda_runtime.h>
#include <starpu.h>
#include <cmath>

extern "C" {

static int cuda_kernel_calls = 0;

__global__ void assign_point_to_cluster_cuda_kernel(
    double *point_values, double *centroids, int K, int dimensions, int *nearestClusterId)
{
    double min_dist = 1e20;
    int bestClusterId = -1;
    for (int i = 0; i < K; i++) {
        double dist = 0.0;
        for (int j = 0; j < dimensions; j++) {
            double diff = centroids[i * dimensions + j] - point_values[j];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            bestClusterId = i;
        }
    }
    *nearestClusterId = bestClusterId + 1;
}

void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg) {
    struct StarPUArgs {
        double *point_values;
        double *centroids;
        int K;
        int dimensions;
        int *nearestClusterId;
    };
    StarPUArgs *args = (StarPUArgs *)cl_arg;
    double *d_point_values, *d_centroids;
    int *d_nearestClusterId;

    cudaMalloc(&d_point_values, args->dimensions * sizeof(double));
    cudaMalloc(&d_centroids, args->K * args->dimensions * sizeof(double));
    cudaMalloc(&d_nearestClusterId, sizeof(int));

    cudaMemcpy(d_point_values, args->point_values, args->dimensions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, args->centroids, args->K * args->dimensions * sizeof(double), cudaMemcpyHostToDevice);

    assign_point_to_cluster_cuda_kernel<<<1,1>>>(d_point_values, d_centroids, args->K, args->dimensions, d_nearestClusterId);

    cudaMemcpy(args->nearestClusterId, d_nearestClusterId, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_point_values);
    cudaFree(d_centroids);
    cudaFree(d_nearestClusterId);

    // Incrementa o contador toda vez que o kernel é chamado
    cuda_kernel_calls++;
}

// Função para acessar o contador
int get_cuda_kernel_calls() {
    return cuda_kernel_calls;
}

}