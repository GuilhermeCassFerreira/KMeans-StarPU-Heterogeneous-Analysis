#include "kmeans_seq.h"
#include <limits>
#include <cstring>

// Implementação da atribuição de pontos a clusters (sequencial)
int assign_point_to_cluster_seq(double *points, double *centroids, int *labels, int n_points, int K, int dimensions) {
    int changes = 0;
    for (int i = 0; i < n_points; i++) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;
        for (int k = 0; k < K; k++) {
            double dist = 0.0;
            for (int d = 0; d < dimensions; d++) {
                double diff = points[i * dimensions + d] - centroids[k * dimensions + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        if (labels[i] != (best_cluster + 1)) {
            labels[i] = best_cluster + 1;
            changes++;
        }
    }
    return changes;
}

// Implementação do cálculo de somas parciais (sequencial)
void calculate_partial_sums_seq(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions) {
    for (int i = 0; i < n_points; i++) {
        int cluster = labels[i] - 1;  // Labels são 1-based
        partial_counts[cluster]++;
        for (int d = 0; d < dimensions; d++) {
            partial_sums[cluster * dimensions + d] += points[i * dimensions + d];
        }
    }
}

// Implementação da atualização de centroides (sequencial)
void update_centroids_seq(double *global_sums, int *global_counts, double *centroids, int K, int dimensions) {
    for (int k = 0; k < K; k++) {
        if (global_counts[k] > 0) {
            for (int d = 0; d < dimensions; d++) {
                centroids[k * dimensions + d] = global_sums[k * dimensions + d] / global_counts[k];
            }
        }
    }
}