#ifndef KMEANS_SEQ_H
#define KMEANS_SEQ_H

#include <vector>
#include "../../include/kmeans_types.h"

// Declarações das funções de kernel para versão sequencial
int assign_point_to_cluster_seq(double *points, double *centroids, int *labels, int n_points, int K, int dimensions);
void calculate_partial_sums_seq(double *points, int *labels, double *partial_sums, int *partial_counts, int n_points, int K, int dimensions);
void update_centroids_seq(double *global_sums, int *global_counts, double *centroids, int K, int dimensions);

#endif // KMEANS_SEQ_H