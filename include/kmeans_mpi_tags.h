#ifndef KMEANS_MPI_TAGS_H
#define KMEANS_MPI_TAGS_H

/**
 * @brief Definições de Tags para Comunicação StarPU-MPI
 *
 * O StarPU exige tags únicas para cada handle registrado no MPI.
 *
 * Versão de paralelismo máximo: cada chunk tem buffers próprios (labels,
 * labels-sombra, contador de mudanças, soma parcial e contagem parcial).
 * Cada família de handles ocupa uma faixa de tags bem separada para nunca
 * colidir, mesmo com num_chunks grande.
 */
namespace KMeansTags {
    // ---- dados globais ----
    const int POINTS        = 10;
    const int LABELS        = 20;
    const int LABELS_PREV   = 30;
    const int CENTROIDS     = 40;
    const int CONVERGED_TAG = 50;
    const int CHANGES_TAG   = 60;

    // ---- acumulador parcial por nó MPI ----
    const int PARTIAL_SUMS_BASE   = 2000;
    const int PARTIAL_COUNTS_BASE = 3000;

    // ---- handles por chunk (faixas bem espaçadas) ----
    const int CHUNK_POINTS_BASE      = 10000000;
    const int CHUNK_LABELS_BASE      = 20000000;
    const int CHUNK_LABELS_PREV_BASE = 30000000;
    const int CHUNK_CHANGES_BASE     = 40000000;
    const int CHUNK_SUMS_BASE        = 50000000;
    const int CHUNK_COUNTS_BASE      = 60000000;
}

#endif