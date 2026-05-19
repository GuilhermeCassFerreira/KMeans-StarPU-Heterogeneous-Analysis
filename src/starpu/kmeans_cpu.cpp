#include "kmeans_runtime.h"
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <iostream>

/* ========================================================================== */
/* Contadores globais (definições)                                            */
/* ========================================================================== */

int cpu_kernel_calls = 0;
int cpu_assign_calls = 0;
int cpu_calculate_calls = 0;
int opencl_assign_calls = 0;
int opencl_calculate_calls = 0;

/* ========================================================================== */
/* TASKS DE NEGÓCIO (CPU) — Ghost tasks via data handle                      */
/*                                                                            */
/* converged_handle é um vetor de 2 inteiros: [flag, iteração].              */
/* Todos os kernels leem [0] (flag) via STARPU_R. Se == 1, return imediato.  */
/* update_centroids escreve os dois via STARPU_RW.                           */
/* ========================================================================== */

/* ASSIGN: buffers[0]=points(R) [1]=centroids(R) [2]=labels(W) [3]=converged(R) */
void assign_point_to_cluster_handles(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_assign_calls++;

    int *conv = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);
    if (conv[0]) return;

    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values  = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids      = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *nearestClusterIds = (int *)   STARPU_VECTOR_GET_PTR(buffers[2]);

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
        nearestClusterIds[idx] = bestClusterId + 1;
    }
}

/* CALCULATE: buffers[0]=points(R) [1]=labels(R) [2]=sums(RW) [3]=counts(RW) [4]=converged(R) */
void calculate_partial_sums(void *buffers[], void *cl_arg) {
    int *conv = (int *)STARPU_VECTOR_GET_PTR(buffers[4]);
    if (conv[0]) return;

    cpu_calculate_calls++;

    int K, dimensions, chunk_size;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &chunk_size);

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

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

/* CLEAN: buffers[0]=sums(W) [1]=counts(W) [2]=converged(R) */
void clean_buffers_cpu(void *buffers[], void *cl_arg) {
    int *conv = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);
    if (conv[0]) return;

    int K, dimensions, dummy_chunk;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &dummy_chunk);

    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);

    std::memset(partial_sums, 0, K * dimensions * sizeof(double));
    std::memset(partial_counts, 0, K * sizeof(int));
}

/* UPDATE CENTROIDS: buffers[0]=sums(R) [1]=counts(R) [2]=centroids(RW) [3]=converged(RW)
 *
 * - Lê centróides antigos (RW no centroids)
 * - Calcula novos centróides
 * - Compara novos vs antigos
 * - Se iguais: conv[0]=1, conv[1]=iteração atual
 * - Se já convergiu antes: mantém e retorna
 *
 * old_centroids é alocado no heap (new/delete) em vez de VLA na stack,
 * para evitar estouro de stack com K*dimensions grande.
 */
void update_centroids_cpu(void *buffers[], void *cl_arg) {
    int *conv = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);
    if (conv[0]) return;

    int K, dimensions, current_iter;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions, &current_iter);

    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int total_values = K * dimensions;

    // Guarda centróides antigos para comparação (heap, não stack)
    double *old_centroids = new double[total_values];
    std::memcpy(old_centroids, centroids, total_values * sizeof(double));

    // Calcula novos centróides
    for (int c = 0; c < K; ++c) {
        if (partial_counts[c] > 0) {
            for (int d = 0; d < dimensions; ++d) {
                centroids[c * dimensions + d] = partial_sums[c * dimensions + d] / partial_counts[c];
            }
        }
    }

    // Compara novos vs antigos
    int changed = 0;
    for (int i = 0; i < total_values; ++i) {
        if (centroids[i] != old_centroids[i]) {
            changed = 1;
            break;
        }
    }

    delete[] old_centroids;

    if (changed == 0) {
        conv[0] = 1;                // flag: convergiu
        conv[1] = current_iter;     // iteração em que convergiu
    }
}

/* ========================================================================== */
/* FUNÇÕES DE REDUÇÃO (CPU)                                                   */
/* ========================================================================== */

void redux_double_init_cpu(void *buffers[], void *cl_arg) {
    double *arr = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);
    std::memset(arr, 0, n * sizeof(double));
}

void redux_double_reduce_cpu(void *buffers[], void *cl_arg) {
    double *dst = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *src = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

void redux_int_init_cpu(void *buffers[], void *cl_arg) {
    int *arr = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);
    std::memset(arr, 0, n * sizeof(int));
}

void redux_int_reduce_cpu(void *buffers[], void *cl_arg) {
    int *dst = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *src = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

/* ACCUMULATE: buffers[0]=sums_dst(RW) [1]=counts_dst(RW) [2]=sums_src(R) [3]=counts_src(R) [4]=converged(R) */
void accumulate_nodes_cpu(void *buffers[], void *cl_arg) {
    int *conv = (int *)STARPU_VECTOR_GET_PTR(buffers[4]);
    if (conv[0]) return;

    int K, dimensions;
    starpu_codelet_unpack_args(cl_arg, &K, &dimensions);

    double *sums_dest = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *counts_dest = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *sums_src = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *counts_src = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    for(int i = 0; i < K * dimensions; i++) sums_dest[i] += sums_src[i];
    for(int i = 0; i < K; i++)              counts_dest[i] += counts_src[i];
}