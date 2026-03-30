#include "kmeans_omp.h"
#include <cmath>
#include <limits>

void assign_points_cpu(double* points, double* centroids, int* labels, int N, int dimensions, int K) {
    // Versão Sequencial: Sem #pragma omp parallel for por enquanto
    for (int i = 0; i < N; i++) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;
        
        for (int c = 0; c < K; c++) {
            double dist = 0.0;
            for (int d = 0; d < dimensions; d++) {
                double diff = points[i * dimensions + d] - centroids[c * dimensions + d];
                dist += diff * diff; // Distância Euclidiana (ao quadrado para ser mais rápido)
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        labels[i] = best_cluster + 1;
    }
}

void calculate_partial_sums_cpu(double* points, int* labels, double* partial_sums, int* partial_counts, int N, int dimensions, int K) {
    // Versão Sequencial: Soma os pontos aos seus respectivos clusters
    for (int i = 0; i < N; i++) {
        int cluster_id = labels[i] - 1;
        if (cluster_id >= 0 && cluster_id < K) {
            partial_counts[cluster_id]++;
            for (int d = 0; d < dimensions; d++) {
                partial_sums[cluster_id * dimensions + d] += points[i * dimensions + d];
            }
        }
    }
}