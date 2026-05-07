#ifndef KMEANS_MPI_TAGS_H
#define KMEANS_MPI_TAGS_H

/**
 * @brief Definições de Tags para Comunicação StarPU-MPI
 * * O StarPU exige tags únicas para cada handle registrado no MPI.
 */
namespace KMeansTags {
    // Tags de Dados Globais
    const int POINTS = 10;   // Registro principal dos pontos
    const int LABELS = 20;   // Registro principal dos rótulos (outputs)
    const int CENTROIDS = 40;   // Registro dos centroides compartilhados

    const int CHUNK_POINTS_BASE = 100000;
    const int CHUNK_LABELS_BASE = 1000000;

    const int PARTIAL_SUMS_BASE = 2000;
    const int PARTIAL_COUNTS_BASE = 3000;
    const int CONVERGED_TAG = 9999; // Tag para a flag de convergência global
}

#endif