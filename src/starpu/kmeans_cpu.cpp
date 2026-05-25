#include "kmeans_runtime.h"
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <iostream>
#include <vector>

/* ========================================================================== */
/* Contadores globais (definições)                                            */
/* ========================================================================== */

int cpu_kernel_calls = 0;
int cpu_assign_calls = 0;
int cpu_calculate_calls = 0;
int cpu_clean_calls = 0;
int cpu_update_calls = 0;
int cpu_accumulate_calls = 0;

int opencl_assign_calls = 0;
int opencl_calculate_calls = 0;
int opencl_clean_calls = 0;
int opencl_update_calls = 0;
int opencl_accumulate_calls = 0;

/* ========================================================================== */
/* TASKS DE NEGÓCIO (CPU) - Com Ghost Tasks (Early Exit)                      */
/* ========================================================================== */

void assign_point_to_cluster_handles(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_assign_calls++;

    int *converged = (int *)STARPU_VARIABLE_GET_PTR(buffers[3]);
    if (*converged == 1) return;

    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values     = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids         = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int    *nearestClusterIds = (int *)   STARPU_VECTOR_GET_PTR(buffers[2]);
    int    *local_changes     = (int *)   STARPU_VARIABLE_GET_PTR(buffers[4]);

    int changes = 0;
    for (int idx = 0; idx < chunk_size; idx++) {
        double *point_values = points_values + idx * dimensions;
        double min_dist2 = std::numeric_limits<double>::max();
        int bestClusterId = -1;
        for (int i = 0; i < K; i++) {
            double dist2 = 0.0;
            for (int j = 0; j < dimensions; j++) {
                double diff = centroids[i * dimensions + j] - point_values[j];
                dist2 += diff * diff;
            }
            if (dist2 < min_dist2) {
                min_dist2 = dist2;
                bestClusterId = i;
            }
        }
        int new_label = bestClusterId + 1;
        if (nearestClusterIds[idx] != new_label) {
            nearestClusterIds[idx] = new_label;
            changes++;
        }
    }
    *local_changes = changes; // REDUX: StarPU soma todas as cópias privadas
}

void calculate_partial_sums(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_calculate_calls++;

    int *converged = (int *)STARPU_VARIABLE_GET_PTR(buffers[4]);
    if (*converged == 1) return; 

    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    /* Zera o buffer exclusivo deste chunk (modo W: não há valor anterior válido) */
    std::memset(partial_sums, 0, (size_t)K * dimensions * sizeof(double));
    std::memset(partial_counts, 0, (size_t)K * sizeof(int));

    for (int idx = 0; idx < chunk_size; ++idx) {
        int cluster_id = nearestClusterIds[idx] - 1;
        if (cluster_id >= 0 && cluster_id < K) {
            partial_counts[cluster_id]++;
            for (int d = 0; d < dimensions; ++d) {
                partial_sums[cluster_id * dimensions + d] += points_values[idx * dimensions + d];
            }
        }
    }
}

void clean_buffers_cpu(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_clean_calls++;

    int *converged = (int *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    if (*converged == 1) return;

    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);

    std::memset(partial_sums, 0, K * dimensions * sizeof(double));
    std::memset(partial_counts, 0, K * sizeof(int));
}

void update_centroids_cpu(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_update_calls++;

    int *converged     = (int *)STARPU_VARIABLE_GET_PTR(buffers[3]);
    if (*converged == 1) return;

    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums   = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int    *partial_counts = (int *)   STARPU_VECTOR_GET_PTR(buffers[1]);
    double *centroids      = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int    *total_changes  = (int *)   STARPU_VARIABLE_GET_PTR(buffers[4]);

    for (int c = 0; c < K; ++c) {
        if (partial_counts[c] > 0) {
            for (int d = 0; d < dimensions; ++d) {
                centroids[c * dimensions + d] =
                    partial_sums[c * dimensions + d] / partial_counts[c];
            }
        }
    }

    // Mesmo critério do OMP: convergência quando zero mudanças de label em todos os chunks
    if (*total_changes == 0) *converged = 1;
    *total_changes = 0; // reseta para a próxima iteração (REDUX parte de 0)
}

void accumulate_nodes_cpu(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_accumulate_calls++;

    int *converged = (int *)STARPU_VARIABLE_GET_PTR(buffers[4]);
    if (*converged == 1) return; 

    int K, dimensions;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions);

    double *sums_dest = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *counts_dest = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *sums_src = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *counts_src = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    for(int i = 0; i < K * dimensions; i++) {
        sums_dest[i] += sums_src[i];
    }
    for(int i = 0; i < K; i++) {
        counts_dest[i] += counts_src[i];
    }
}
/* ========================================================================== */
/* REDUX CPU — init: zera cópia privada; reduce: merge de duas cópias        */
/* ========================================================================== */

void redux_double_init_cpu(void *buffers[], void *) {
    double *arr = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = (int)STARPU_VECTOR_GET_NX(buffers[0]);
    std::memset(arr, 0, (size_t)n * sizeof(double));
}

void redux_double_reduce_cpu(void *buffers[], void *) {
    double *dst = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    const double *src = (const double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = (int)STARPU_VECTOR_GET_NX(buffers[0]);
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

void redux_int_init_cpu(void *buffers[], void *) {
    int *arr = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = (int)STARPU_VECTOR_GET_NX(buffers[0]);
    std::memset(arr, 0, (size_t)n * sizeof(int));
}

void redux_int_reduce_cpu(void *buffers[], void *) {
    int *dst = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    const int *src = (const int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = (int)STARPU_VECTOR_GET_NX(buffers[0]);
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

/* REDUX para h_changes (VARIABLE = scalar int) */
void changes_var_init_cpu(void *buffers[], void *) {
    int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    *val = 0;
}

void changes_var_reduce_cpu(void *buffers[], void *) {
    int *dst       = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const int *src = (const int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    *dst += *src;
}
