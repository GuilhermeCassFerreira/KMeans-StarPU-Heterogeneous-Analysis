#include "kmeans_runtime.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iomanip> 
#include <fstream> 
#include "kmeans_mpi_tags.h"

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
/* Codelets StarPU (definições atualizadas para Ghost Tasks)                  */
/* ========================================================================== */

struct starpu_codelet cl_assign_point_handles = {
    .cpu_funcs = {assign_point_to_cluster_handles},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {assign_point_to_cluster_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 4, // 3 buffers de dados + 1 flag de convergência
    .modes = {STARPU_R, STARPU_R, STARPU_RW, STARPU_R},
    .model = &assign_perf_model
};

struct starpu_codelet cl_calculate_partial_sums = {
    .cpu_funcs = {calculate_partial_sums},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {calculate_partial_sums_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 5, // 4 buffers de dados + 1 flag de convergência
    .modes = {STARPU_R, STARPU_R, STARPU_RW, STARPU_RW, STARPU_R},
    .model = &calculate_perf_model
};

struct starpu_codelet cl_clean_buffers = {
    .cpu_funcs = {clean_buffers_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {clean_buffers_cuda}, 
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 3, // 2 buffers de dados + 1 flag de convergência
    .modes = {STARPU_W, STARPU_W, STARPU_R},
    .model = &clean_perf_model
};

struct starpu_codelet cl_update_centroids = {
    .cpu_funcs = {update_centroids_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {update_centroids_cuda}, 
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 4, // 3 buffers de dados + 1 flag de convergência
    .modes = {STARPU_R, STARPU_R, STARPU_RW, STARPU_RW}, // Centroids agora são RW
    .model = &update_perf_model
};

struct starpu_codelet cl_accumulate_nodes = {
    .cpu_funcs = {accumulate_nodes_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {accumulate_nodes_cuda}, 
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 5, // 4 buffers de dados + 1 flag de convergência
    .modes = {STARPU_RW, STARPU_RW, STARPU_R, STARPU_R, STARPU_R},
    .name = "kmeans_accumulate_mpi"
};

/* ========================================================================== */
/* Implementação da classe KMeans                                             */
/* ========================================================================== */

KMeans::KMeans(int K, int iterations, string output_dir, int chunk_size, int rank, int size, int dims, int seed)
    : K(K), iters(iterations), output_dir(output_dir), chunk_size(chunk_size), mpi_rank(rank),
      world_size(size), dimensions(dims), seed(seed),
      points_handle(nullptr), output_handle(nullptr),
      num_chunks(0), partial_sums_ptr(nullptr), partial_counts_ptr(nullptr),
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

void KMeans::assignPointsToClusters(int N, starpu_data_handle_t converged_handle) {
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
        if (this_chunk <= 0) break;

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_assign_point_handles,
            STARPU_R, points_children[chunk_id],
            STARPU_R, centroids_handle,
            STARPU_RW, outputs_children[chunk_id],
            STARPU_R, converged_handle, // Passando a Flag
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &this_chunk, sizeof(int),
            0);
    }
}

void KMeans::calculateCentroids(int N, starpu_data_handle_t converged_handle) {
    int dummy_chunk = 0;

    for (int n = 0; n < world_size; n++) {
        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_clean_buffers,
            STARPU_W, partial_sums_handle[n],
            STARPU_W, partial_counts_handle[n],
            STARPU_R, converged_handle, // Passando a Flag
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &dummy_chunk, sizeof(int),
            STARPU_EXECUTE_ON_NODE, n,
            0);
    }
    
    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        int owners = chunk_owners[chunk_id];
        int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
        if (this_chunk <= 0) break;

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_calculate_partial_sums,
            STARPU_R, points_children[chunk_id],
            STARPU_R, outputs_children[chunk_id],
            STARPU_RW, partial_sums_handle[owners],
            STARPU_RW, partial_counts_handle[owners],
            STARPU_R, converged_handle, // Passando a Flag
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &this_chunk, sizeof(int),
            0);
    }

    reduceCentroidsAcrossNodes(converged_handle);

    starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_update_centroids,
        STARPU_R, partial_sums_handle[0],
        STARPU_R, partial_counts_handle[0],
        STARPU_RW, centroids_handle, // RW para poder ler o velho e checar se mudou
        STARPU_RW, converged_handle, // RW para o Nodo 0 poder alterar para 1 se convergir
        STARPU_VALUE, &K, sizeof(int),
        STARPU_VALUE, &dimensions, sizeof(int),
        STARPU_VALUE, &dummy_chunk, sizeof(int),
        STARPU_EXECUTE_ON_NODE, 0,
        0);
}

void KMeans::reduceCentroidsAcrossNodes(starpu_data_handle_t converged_handle) {
    if (world_size <= 1) return;

    for (int n = 1; n < world_size; n++) {
        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_accumulate_nodes,
            STARPU_RW, partial_sums_handle[0],   
            STARPU_RW, partial_counts_handle[0], 
            STARPU_R, partial_sums_handle[n],    
            STARPU_R, partial_counts_handle[n],  
            STARPU_R, converged_handle, // Passando a Flag
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_EXECUTE_ON_NODE, 0,    
            0);
    }
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
            memset(labels_ptr, 0, labels_bytes);
        }
    } else {
        memset(points_ptr, 0, points_bytes);
        memset(labels_ptr, 0, labels_bytes);
    }
    memset(partial_sums_ptr, 0, sums_bytes);
    memset(partial_counts_ptr, 0, counts_bytes);

    starpu_vector_data_register(&points_handle, STARPU_MAIN_RAM, (uintptr_t)points_ptr, N, dimensions * sizeof(double));
    starpu_vector_data_register(&output_handle, STARPU_MAIN_RAM, (uintptr_t)labels_ptr, N, sizeof(int));

    starpu_mpi_data_register(points_handle, KMeansTags::POINTS, 0);
    starpu_mpi_data_register(output_handle, KMeansTags::LABELS, 0);

    partial_sums_handle.resize(world_size);
    partial_counts_handle.resize(world_size);

    for (int n = 0; n < world_size; n++) {
        if (n == mpi_rank) {
            starpu_vector_data_register(&partial_sums_handle[n], STARPU_MAIN_RAM, (uintptr_t)partial_sums_ptr, K * dimensions, sizeof(double));
            starpu_vector_data_register(&partial_counts_handle[n], STARPU_MAIN_RAM, (uintptr_t)partial_counts_ptr, K, sizeof(int));
        } else {
            starpu_vector_data_register(&partial_sums_handle[n], -1, (uintptr_t)NULL, K * dimensions, sizeof(double));
            starpu_vector_data_register(&partial_counts_handle[n], -1, (uintptr_t)NULL, K, sizeof(int));
        }
        starpu_mpi_data_register(partial_sums_handle[n], KMeansTags::PARTIAL_SUMS_BASE + n, n);
        starpu_mpi_data_register(partial_counts_handle[n], KMeansTags::PARTIAL_COUNTS_BASE + n, n);
    }

    num_chunks = (N + this->chunk_size - 1) / this->chunk_size;
    struct starpu_data_filter filterChunks = {
        .filter_func = starpu_vector_filter_block,
        .nchildren = (unsigned)num_chunks
    };

    starpu_data_partition(points_handle, &filterChunks);
    starpu_data_partition(output_handle, &filterChunks);

    chunk_owners.resize(num_chunks);
    points_children.resize(num_chunks);
    outputs_children.resize(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        chunk_owners[i] = getChunkOwner(i);

        points_children[i] = starpu_data_get_child(points_handle, i);
        outputs_children[i] = starpu_data_get_child(output_handle, i);

        starpu_mpi_data_register(points_children[i], KMeansTags::CHUNK_POINTS_BASE + i, 0);
        starpu_mpi_data_register(outputs_children[i], KMeansTags::CHUNK_LABELS_BASE + i, 0);
    }

    centroids_data.resize(K * dimensions);

    if (mpi_rank == 0) {
        srand(this->seed);

        std::vector<int> chosen_indices;
        
        while ((int)chosen_indices.size() < K) {
            int r = rand() % N; 
            if (std::find(chosen_indices.begin(), chosen_indices.end(), r) == chosen_indices.end()) {
                chosen_indices.push_back(r);
            }
        }

        for (int i = 0; i < K; ++i) {
            int point_idx = chosen_indices[i];
            
            clusters.emplace_back(i + 1, all_points[point_idx]);
            
            for (int j = 0; j < dimensions; j++) {
                centroids_data[i * dimensions + j] = all_points[point_idx].getVal(j);
            }
        }
        
        cout << "[INFO] Centroides iniciais escolhidos aleatoriamente (Seed: " << this->seed << ")" << endl;
    }

    MPI_Bcast(centroids_data.data(), K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    starpu_vector_data_register(&centroids_handle, STARPU_MAIN_RAM, (uintptr_t)centroids_data.data(), K * dimensions, sizeof(double));
    starpu_mpi_data_register(centroids_handle, KMeansTags::CENTROIDS, 0);

    if (mpi_rank == 0) cout << "[INFO] Injetando max " << iters << " iteracoes no DAG (Assíncrono)..." << endl;

    // --- REGISTRO DA FLAG DE CONVERGÊNCIA ---
    int converged_flag = 0; 
    starpu_data_handle_t converged_handle;
    starpu_variable_data_register(&converged_handle, STARPU_MAIN_RAM, (uintptr_t)&converged_flag, sizeof(int));
    // Certifique-se que KMeansTags::CONVERGED_TAG existe no seu kmeans_mpi_tags.h!
    starpu_mpi_data_register(converged_handle, KMeansTags::CONVERGED_TAG, 0);

    for (int it = 0; it < iters; ++it) {
        assignPointsToClusters(N, converged_handle);
        calculateCentroids(N, converged_handle);
    }

    // Aguarda o término de todas as tarefas assíncronas
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        if (converged_flag) {
            cout << "[INFO] Early Exit concluído com sucesso." << endl;
        } else {
            cout << "[INFO] Limite de iterações atingido sem convergência completa." << endl;
        }
    }

    starpu_data_unregister(converged_handle); // Desregistra a flag!

    // Garante que o Nodo 0 detenha a versão mais recente dos labels
    for (int i = 0; i < num_chunks; i++) {
        starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD, outputs_children[i], 0, NULL, NULL);
    }
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    // Remove o particionamento
    starpu_data_unpartition(points_handle, STARPU_MAIN_RAM);
    starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);

    if (mpi_rank == 0) {
        starpu_data_acquire(output_handle, STARPU_R);
        for (int i = 0; i < N; i++) {
            all_points[i].setCluster(labels_ptr[i]);
        }
        starpu_data_release(output_handle);
        
        cout << "[INFO] Salvando centróides finais..." << endl;

        string cmd = "mkdir -p " + output_dir;
        if (system(cmd.c_str()) != 0) {
            cerr << "[AVISO] Falha ao tentar criar o diretório: " << output_dir << endl;
        }

        string clusters_filename = output_dir + "/" + to_string(K) + "-clusters.txt";
        ofstream outfile(clusters_filename);

        if (!outfile.is_open()) {
            cerr << "[ERRO] Não foi possível criar o arquivo: " << clusters_filename << endl;
        } else {
            outfile << fixed << setprecision(6);
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < dimensions; j++) {
                    double val = centroids_data[i * dimensions + j];
                    outfile << val << " ";
                }
                outfile << endl;
            }
            outfile.close();
            cout << "[INFO] Arquivo de centróides salvo com sucesso em: " << clusters_filename << endl;
        }
    }

    // Limpeza de Memória e Handles
    starpu_data_unregister(points_handle);
    starpu_data_unregister(output_handle);
    starpu_data_unregister(centroids_handle);
    for (int n = 0; n < world_size; n++) {
        starpu_data_unregister(partial_sums_handle[n]);
        starpu_data_unregister(partial_counts_handle[n]);
    }

    starpu_free_noflag(points_ptr, points_bytes);
    starpu_free_noflag(labels_ptr, labels_bytes);
    starpu_free_noflag(partial_sums_ptr, sums_bytes);
    starpu_free_noflag(partial_counts_ptr, counts_bytes);
}