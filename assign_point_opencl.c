#include <starpu.h>
#include <starpu_opencl.h>
#include <stdio.h>
#include <math.h>

static int opencl_kernel_calls = 0;

// Garante linkage C para uso com C++
#ifdef __cplusplus
extern "C" {
#endif

void assign_point_to_cluster_opencl(void *buffers[], void *cl_arg)
{
    struct StarPUArgs {
        double *points_values;      // [chunk_size][dimensions] (flattened)
        int *nearestClusterIds;     // [chunk_size]
        double *centroids;          // [K][dimensions] (flattened)
        int K;
        int dimensions;
        int offset;
        int chunk_size;
    };
    struct StarPUArgs *args = (struct StarPUArgs *)cl_arg;

    if (!args) return;

    double *points_values = args->points_values;
    double *centroids = args->centroids;
    int K = args->K;
    int dimensions = args->dimensions;
    int chunk_size = args->chunk_size;
    int *nearestClusterIds = args->nearestClusterIds;

    // Checagem defensiva
    if (!points_values || !centroids || !nearestClusterIds || chunk_size <= 0 || K <= 0 || dimensions <= 0) {
        fprintf(stderr, "[OpenCL-host] Args inválidos\n");
        return;
    }

    for (int idx = 0; idx < chunk_size; idx++) {
        double *point_values = points_values + idx * dimensions;
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
        nearestClusterIds[idx] = bestClusterId + 1;
    }

    opencl_kernel_calls++;
}

int get_opencl_kernel_calls() {
    return opencl_kernel_calls;
}

#ifdef __cplusplus
}
#endif