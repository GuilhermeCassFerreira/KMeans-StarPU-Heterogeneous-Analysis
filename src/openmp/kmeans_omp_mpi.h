#ifndef KMEANS_OMP_MPI_H
#define KMEANS_OMP_MPI_H

#include <algorithm>

// Contadores para verificação de offload
extern int cuda_assign_calls;
extern int cuda_calculate_calls;

// Definição dos tipos de função para os ponteiros
typedef void (*assign_fn)(double*, double*, int*, int, int, int);
typedef void (*calculate_fn)(double*, int*, double*, int*, int, int, int);

// Funções CPU
void assign_point_to_cluster_cpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions);
void calculate_partial_sums_cpu(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions);

// Funções GPU
void assign_point_to_cluster_gpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions);
void calculate_partial_sums_gpu(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions);

#endif