#include "kmeans_runtime.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iomanip>

using namespace std;

/* ========================================================================== */
/* Performance Models (definições)                                            */
/* ========================================================================== */

struct starpu_perfmodel assign_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_assign_model_mpi"
};

struct starpu_perfmodel calculate_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_calculate_model_mpi"
};

struct starpu_perfmodel clean_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_clean_model_mpi"
};

struct starpu_perfmodel update_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_update_model_mpi"
};

/* ========================================================================== */
/* Codelets StarPU (definições)                                               */
/* ========================================================================== */

struct starpu_codelet cl_assign_point_handles = {
    .cpu_funcs = {assign_point_to_cluster_handles},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {assign_point_to_cluster_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .model = &assign_perf_model
};

struct starpu_codelet cl_calculate_partial_sums = {
    .cpu_funcs = {calculate_partial_sums},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {calculate_partial_sums_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 4,
    .modes = {STARPU_R, STARPU_R, STARPU_RW, STARPU_RW},
    .model = &calculate_perf_model
};

struct starpu_codelet cl_clean_buffers = {
    .cpu_funcs = {clean_buffers_cpu},
    .nbuffers = 2,
    .modes = {STARPU_W, STARPU_W},
    .model = &clean_perf_model
};

struct starpu_codelet cl_update_centroids = {
    .cpu_funcs = {update_centroids_cpu},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .model = &update_perf_model
};

/* ========================================================================== */
/* Implementação da classe KMeans                                             */
/* ========================================================================== */

KMeans::KMeans(int K, int iterations, string output_dir, int chunk_size,
               bool use_heterogeneous_chunks, int rank, int size, int dims)
    : K(K), iters(iterations), output_dir(output_dir), chunk_size(chunk_size),
      use_heterogeneous_chunks(use_heterogeneous_chunks), mpi_rank(rank),
      world_size(size), dimensions(dims),
      points_handle(nullptr), output_handle(nullptr),
      num_chunks(0), partial_sums_ptr(nullptr), partial_counts_ptr(nullptr),
      partial_sums_handle(nullptr), partial_counts_handle(nullptr),
      centroids_handle(nullptr), points_ptr(nullptr), labels_ptr(nullptr),
      total_points(0)
{
}

void KMeans::clearClusters() {
    for (int i = 0; i < K; i++) {
        clusters[i].removeAllPoints();
    }
}

int KMeans::getChunkOwner(int chunk_id) {
    return chunk_id % world_size;
}

void KMeans::assignPointsToClusters(int N) {
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
        if (this_chunk <= 0) break;

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_assign_point_handles,
            STARPU_R, points_children[chunk_id],
            STARPU_R, centroids_handle,
            STARPU_W, outputs_children[chunk_id],
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &this_chunk, sizeof(int),
            STARPU_EXECUTE_ON_NODE, chunk_owners[chunk_id],
            0);
    }
}

void KMeans::calculateCentroids(int N) {
    int dummy_chunk = 0;

    starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_clean_buffers,
        STARPU_W, partial_sums_handle,
        STARPU_W, partial_counts_handle,
        STARPU_VALUE, &K, sizeof(int),
        STARPU_VALUE, &dimensions, sizeof(int),
        STARPU_VALUE, &dummy_chunk, sizeof(int),
        STARPU_EXECUTE_ON_NODE, mpi_rank,
        0);

    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        if (chunk_owners[chunk_id] != mpi_rank) continue;

        int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
        if (this_chunk <= 0) break;

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_calculate_partial_sums,
            STARPU_R, points_children[chunk_id],
            STARPU_R, outputs_children[chunk_id],
            STARPU_RW, partial_sums_handle,
            STARPU_RW, partial_counts_handle,
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &this_chunk, sizeof(int),
            STARPU_EXECUTE_ON_NODE, chunk_owners[chunk_id],
            0);
    }

    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    reduceCentroidsAcrossNodes();

    starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_update_centroids,
        STARPU_R, partial_sums_handle,
        STARPU_R, partial_counts_handle,
        STARPU_W, centroids_handle,
        STARPU_VALUE, &K, sizeof(int),
        STARPU_VALUE, &dimensions, sizeof(int),
        STARPU_VALUE, &dummy_chunk, sizeof(int),
        STARPU_EXECUTE_ON_NODE, 0,
        0);
}

void KMeans::reduceCentroidsAcrossNodes() {
    if (world_size <= 1) return;

    size_t sums_count = K * dimensions;
    size_t counts_count = K;

    starpu_data_acquire(partial_sums_handle, STARPU_RW);
    starpu_data_acquire(partial_counts_handle, STARPU_RW);

    double *local_sums = (double *)starpu_vector_get_local_ptr(partial_sums_handle);
    int *local_counts = (int *)starpu_vector_get_local_ptr(partial_counts_handle);

    vector<double> reduced_sums(sums_count, 0.0);
    vector<int> reduced_counts(counts_count, 0);

    MPI_Allreduce(local_sums, reduced_sums.data(), sums_count,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_counts, reduced_counts.data(), counts_count,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    memcpy(local_sums, reduced_sums.data(), sums_count * sizeof(double));
    memcpy(local_counts, reduced_counts.data(), counts_count * sizeof(int));

    starpu_data_release(partial_counts_handle);
    starpu_data_release(partial_sums_handle);
}

void KMeans::run(vector<Point> &all_points, int N) {
    total_points = N;
    points_ptr = nullptr;
    labels_ptr = nullptr;
    partial_sums_ptr = nullptr;
    partial_counts_ptr = nullptr;

    size_t points_bytes = (size_t)N * dimensions * sizeof(double);
    size_t labels_bytes = (size_t)N * sizeof(int);
    size_t sums_bytes   = (size_t)K * dimensions * sizeof(double);
    size_t counts_bytes = (size_t)K * sizeof(int);

    if (starpu_malloc((void**)&points_ptr, points_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&labels_ptr, labels_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&partial_sums_ptr, sums_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&partial_counts_ptr, counts_bytes) != 0) exit(1);

    if (mpi_rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < dimensions; d++)
                points_ptr[i * dimensions + d] = all_points[i].getVal(d);
            labels_ptr[i] = 0;
        }
    } else {
        memset(points_ptr, 0, points_bytes);
        memset(labels_ptr, 0, labels_bytes);
    }
    memset(partial_sums_ptr, 0, sums_bytes);
    memset(partial_counts_ptr, 0, counts_bytes);

    // Registrar handles StarPU
    starpu_vector_data_register(&points_handle, STARPU_MAIN_RAM, (uintptr_t)points_ptr, N, dimensions * sizeof(double));
    starpu_vector_data_register(&output_handle, STARPU_MAIN_RAM, (uintptr_t)labels_ptr, N, sizeof(int));
    starpu_vector_data_register(&partial_sums_handle, STARPU_MAIN_RAM, (uintptr_t)partial_sums_ptr, K * dimensions, sizeof(double));
    starpu_vector_data_register(&partial_counts_handle, STARPU_MAIN_RAM, (uintptr_t)partial_counts_ptr, K, sizeof(int));

    starpu_mpi_data_register(points_handle, 0, 0);
    starpu_mpi_data_register(output_handle, 1, 0);
    starpu_mpi_data_register(partial_sums_handle, 2000 + mpi_rank, mpi_rank);
    starpu_mpi_data_register(partial_counts_handle, 3000 + mpi_rank, mpi_rank);

    // Particionar pontos e labels em chunks
    num_chunks = (N + this->chunk_size - 1) / this->chunk_size;
    struct starpu_data_filter f = {
        .filter_func = starpu_vector_filter_block,
        .nchildren = (unsigned)num_chunks
    };

    starpu_data_partition(points_handle, &f);
    starpu_data_partition(output_handle, &f);

    // Distribuir chunks round-robin entre nodos
    chunk_owners.resize(num_chunks);
    points_children.resize(num_chunks);
    outputs_children.resize(num_chunks);

    vector<int> chunks_per_node(world_size, 0);

    for (int i = 0; i < num_chunks; ++i) {
        chunk_owners[i] = getChunkOwner(i);
        chunks_per_node[chunk_owners[i]]++;

        points_children[i] = starpu_data_get_child(points_handle, i);
        outputs_children[i] = starpu_data_get_child(output_handle, i);

        starpu_mpi_data_register(points_children[i], 100 + i, chunk_owners[i]);
        starpu_mpi_data_register(outputs_children[i], 1000 + i, chunk_owners[i]);
    }

    if (mpi_rank == 0) {
        cout << "[MPI-DIST] Distribuicao de chunks entre " << world_size << " nodos:" << endl;
        for (int n = 0; n < world_size; n++) {
            cout << "  Nodo " << n << ": " << chunks_per_node[n] << " chunks" << endl;
        }
    }

    // Inicializar centroides
    centroids_data.resize(K * dimensions);
    if (mpi_rank == 0) {
        for (int i = 0; i < K; ++i) {
            clusters.emplace_back(i + 1, all_points[i]);
            for (int j = 0; j < dimensions; j++)
                centroids_data[i * dimensions + j] = clusters[i].getCentroidByPos(j);
        }
    }
    MPI_Bcast(centroids_data.data(), K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    starpu_vector_data_register(&centroids_handle, STARPU_MAIN_RAM, (uintptr_t)centroids_data.data(), K * dimensions, sizeof(double));
    starpu_mpi_data_register(centroids_handle, 4, 0);

    if (mpi_rank == 0) cout << "[INFO] Rodando " << iters << " iteracoes..." << endl;

    for (int it = 0; it < iters; ++it) {
        assignPointsToClusters(N);
        calculateCentroids(N);
    }

    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    // Coletar resultados
    if (mpi_rank == 0) {
        starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, centroids_handle, 0, NULL, NULL);
        for (int i = 0; i < num_chunks; ++i)
            starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, outputs_children[i], 0, NULL, NULL);
    }

    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        starpu_data_unpartition(points_handle, STARPU_MAIN_RAM);
        starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);
    } else {
        starpu_data_unpartition(points_handle, -1);
        starpu_data_unpartition(output_handle, -1);
    }

    if (mpi_rank == 0) {
        starpu_data_acquire(output_handle, STARPU_R);
        for (int i = 0; i < N; i++) all_points[i].setCluster(labels_ptr[i]);
        starpu_data_release(output_handle);
    }

    starpu_data_unregister(points_handle);
    starpu_data_unregister(output_handle);
    starpu_data_unregister(centroids_handle);
    starpu_data_unregister(partial_sums_handle);
    starpu_data_unregister(partial_counts_handle);

    starpu_free_noflag(points_ptr, points_bytes);
    starpu_free_noflag(labels_ptr, labels_bytes);
    starpu_free_noflag(partial_sums_ptr, sums_bytes);
    starpu_free_noflag(partial_counts_ptr, counts_bytes);
}