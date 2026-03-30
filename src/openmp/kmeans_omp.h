#ifndef KMEANS_OMP_H
#define KMEANS_OMP_H

#include "../../include/kmeans_types.h"
#include <vector>
#include <string>

// Funções computacionais (Por enquanto, sequenciais)
void assign_points_cpu(double* points, double* centroids, int* labels, int N, int dimensions, int K);
void calculate_partial_sums_cpu(double* points, int* labels, double* partial_sums, int* partial_counts, int N, int dimensions, int K);

// A Classe Maestro
class KMeansOMP {
private:
    int K, iters, dimensions;
    std::string output_dir;
    std::vector<double> centroids_data;

public:
    KMeansOMP(int K, int iterations, std::string output_dir, int dims);
    void run(std::vector<Point> &all_points, int N);
};

#endif