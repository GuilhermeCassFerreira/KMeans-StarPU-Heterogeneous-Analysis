#include <starpu.h>
#include <starpu_opencl.h>
#include <stdio.h>
#include <math.h>

static int opencl_kernel_calls = 0;

void assign_point_to_cluster_opencl(void *buffers[], void *cl_arg)
{
    struct StarPUArgs {
        double *point_values;
        double *centroids;
        int K;
        int dimensions;
        int *nearestClusterId;
    };
    struct StarPUArgs *args = (struct StarPUArgs *)cl_arg;

    // Implementação simples do kernel em C (simulando OpenCL)
    double min_dist = 1e20;
    int bestClusterId = -1;
    for (int i = 0; i < args->K; i++) {
        double dist = 0.0;
        for (int j = 0; j < args->dimensions; j++) {
            double diff = args->centroids[i * args->dimensions + j] - args->point_values[j];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            bestClusterId = i;
        }
    }
    *(args->nearestClusterId) = bestClusterId + 1;

    opencl_kernel_calls++;
}

int get_opencl_kernel_calls() {
    return opencl_kernel_calls;
}