#include <omp.h>
#include <stdio.h>
#include "kmeans_omp_mpi.h"

// Variáveis inicializadas com 0 (isso resolve o erro "undefined reference" no Linker)
int cuda_assign_calls = 0;
int cuda_calculate_calls = 0;

void assign_point_to_cluster_gpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions) {
    int on_device = 0;

    #pragma omp target teams distribute parallel for \
        map(to: points[0:n_points*dimensions], centroids[0:K*dimensions]) \
        map(from: labels[0:n_points], on_device) thread_limit(256)
    for (int i = 0; i < n_points; i++) {
        // Apenas a primeira thread do primeiro bloco faz a checagem para evitar overhead
        if (i == 0) {
            on_device = !omp_is_initial_device();
        }

        double min_dist = 1e300;
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
        labels[i] = best_cluster + 1;
    }

    // Se o flag on_device for 1, incrementamos o contador de chamadas de GPU
    if (on_device) {
        cuda_assign_calls++;
    } else {
        // Opcional: imprimir um aviso se cair na CPU para você depurar
        // printf("[DEBUG] Alerta: assign_point_to_cluster_gpu rodou na CPU!\n");
    }
}

void calculate_partial_sums_gpu(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions) {
    int on_device = 0;

    #pragma omp target teams distribute parallel for \
        map(to: points[0:n_points*dimensions], labels[0:n_points]) \
        map(tofrom: partial_sums[0:K*dimensions], partial_counts[0:K]) \
        map(from: on_device) thread_limit(256)
    for (int i = 0; i < n_points; i++) {
        // Apenas a primeira thread faz a checagem
        if (i == 0) {
            on_device = !omp_is_initial_device();
        }

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

    if (on_device) {
        cuda_calculate_calls++;
    }
}