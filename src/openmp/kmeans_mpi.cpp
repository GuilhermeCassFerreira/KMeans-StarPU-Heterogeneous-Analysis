#include "kmeans_omp.h"
#include <iostream>
#include <cstring>

KMeansOMP::KMeansOMP(int K, int iterations, std::string output_dir, int dims)
    : K(K), iters(iterations), dimensions(dims), output_dir(output_dir) {}

void KMeansOMP::run(std::vector<Point> &all_points, int N) {
    // 1. Converter o std::vector<Point> do C++ para Arrays Contíguos de C
    // (A GPU e o MPI exigem memória perfeitamente contígua)
    double *points_flat = new double[N * dimensions];
    int *labels_flat = new int[N];

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dimensions; d++) {
            points_flat[i * dimensions + d] = all_points[i].getVal(d);
        }
        labels_flat[i] = 0;
    }

    // 2. Inicializar centroides
    centroids_data.resize(K * dimensions);
    for (int i = 0; i < K; ++i) {
        for (int d = 0; d < dimensions; d++) {
            centroids_data[i * dimensions + d] = all_points[i].getVal(d);
        }
    }

    double *sums = new double[K * dimensions];
    int *counts = new int[K];

    std::cout << "[INFO] Rodando K-Means Sequencial por " << iters << " iteracoes..." << std::endl;

    // 3. Loop do K-Means
    for (int it = 0; it < iters; ++it) {
        std::memset(sums, 0, K * dimensions * sizeof(double));
        std::memset(counts, 0, K * sizeof(int));

        assign_points_cpu(points_flat, centroids_data.data(), labels_flat, N, dimensions, K);
        calculate_partial_sums_cpu(points_flat, labels_flat, sums, counts, N, dimensions, K);

        // Atualizar os centroides
        for (int c = 0; c < K; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < dimensions; ++d) {
                    centroids_data[c * dimensions + d] = sums[c * dimensions + d] / counts[c];
                }
            }
        }
    }

    // 4. Salvar resultados de volta nos objetos Point
    for (int i = 0; i < N; i++) {
        all_points[i].setCluster(labels_flat[i]);
    }

    delete[] points_flat;
    delete[] labels_flat;
    delete[] sums;
    delete[] counts;
}