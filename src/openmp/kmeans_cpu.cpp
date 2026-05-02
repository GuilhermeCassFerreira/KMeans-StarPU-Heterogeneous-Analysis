#include <omp.h>
#include <cfloat>
#include "kmeans_omp_mpi.h"

int assign_point_to_cluster_cpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions) {
    int changes = 0;
    #pragma omp parallel for reduction(+:changes)
    for (int i = 0; i < n_points; i++) {
        double min_dist = DBL_MAX;
        int best_cluster = -1;
        int old_label = labels[i];

        for (int k = 0; k < K; k++) {
            double dist = 0;
            for (int d = 0; d < dimensions; d++) {
                double diff = points[i * dimensions + d] - centroids[k * dimensions + d];
                dist += diff * diff;
            }
            if (dist < min_dist) { min_dist = dist; best_cluster = k; }
        }
        
        int new_label = best_cluster + 1;
        if (new_label != old_label) {
            changes++;
            labels[i] = new_label;
        }
    }
    return changes;
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

void update_centroids_cpu(double *global_sums, int *global_counts, double *centroids, int K, int dimensions) {
    #pragma omp parallel for
    for (int k = 0; k < K; k++) {
        if (global_counts[k] > 0) {
            for (int d = 0; d < dimensions; d++) {
                centroids[k * dimensions + d] = global_sums[k * dimensions + d] / global_counts[k];
            }
        }
    }
}