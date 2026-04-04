#include <omp.h>
#include <cfloat>
#include "kmeans_omp_mpi.h"

void assign_point_to_cluster_cpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions) {
    #pragma omp parallel for
    for (int i = 0; i < n_points; i++) {
        double min_dist = DBL_MAX;
        int best_cluster = -1;

        for (int k = 0; k < K; k++) {
            double dist = 0;
            for (int d = 0; d < dimensions; d++) {
                double diff = points[i * dimensions + d] - centroids[k * dimensions + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        labels[i] = best_cluster + 1; // Para bater com o StarPU (1-indexed)
    }
}

void calculate_partial_sums_cpu(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions) {
    #pragma omp parallel for
    for (int i = 0; i < n_points; i++) {
        int cluster_id = labels[i] - 1;
        if (cluster_id >= 0 && cluster_id < K) {
            #pragma omp atomic
            partial_counts[cluster_id]++;
            
            for (int d = 0; d < dimensions; d++) {
                #pragma omp atomic
                partial_sums[cluster_id * dimensions + d] += points[i * dimensions + d];
            }
        }
    }
}