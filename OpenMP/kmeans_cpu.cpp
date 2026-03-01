#include <omp.h>
#include <vector>
#include <cmath>

void assign_clusters_cpu(const double* points, double* centroids, int* labels, int N, int K, int D, int* changed) {
    int local_changes = 0;

    #pragma omp parallel for reduction(+:local_changes)
    for (int i = 0; i < N; i++) {
        double min_dist_sq = 1e30;
        int best_cluster = 0;

        for (int k = 0; k < K; k++) {
            double dist_sq = 0.0;
            for (int d = 0; d < D; d++) {
                double diff = points[i * D + d] - centroids[k * D + d];
                dist_sq += diff * diff;
            }
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_cluster = k;
            }
        }

        if (labels[i] != best_cluster) {
            local_changes++;
            labels[i] = best_cluster;
        }
    }
    *changed = local_changes;
}

void update_centroids_cpu(const double* points, double* centroids, const int* labels, int N, int K, int D) {
    std::vector<double> sum(K * D, 0.0);
    std::vector<int> counts(K, 0);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int c = labels[i];
        #pragma omp atomic
        counts[c]++;
        
        for (int d = 0; d < D; d++) {
            #pragma omp atomic
            sum[c * D + d] += points[i * D + d];
        }
    }

    #pragma omp parallel for
    for (int k = 0; k < K; k++) {
        if (counts[k] > 0) {
            for (int d = 0; d < D; d++) {
                centroids[k * D + d] = sum[k * D + d] / counts[k];
            }
        }
    }
}