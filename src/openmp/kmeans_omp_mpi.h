#ifndef KMEANS_OMP_MPI_H
#define KMEANS_OMP_MPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Ponteiros de função (O Contrato)
typedef void (*assign_fn)(double *points, double *centroids, int *labels, int n_points, int K, int dimensions);
typedef void (*calculate_fn)(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions);

// Funções Multi-core (CPU)
extern void assign_point_to_cluster_cpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions);
extern void calculate_partial_sums_cpu(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions);

// Funções Target Offload (GPU)
extern void assign_point_to_cluster_gpu(double *points, double *centroids, int *labels, int n_points, int K, int dimensions);
extern void calculate_partial_sums_gpu(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions);

#ifdef __cplusplus
}
#endif

#endif // KMEANS_OMP_MPI_H