#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "../../include/kmeans_types.h"
#include "kmeans_omp_mpi.h"

using namespace std;
using namespace chrono;

extern bool read_points_from_file(const string& filename, vector<Point>& points, int& N, int& dimensions);

int main(int argc, char **argv) {
    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("[RANK %d/%d] Operando no host: %s\n", rank, size, processor_name);

    // Verificação de Argumentos (Atualizado com Modo 3 e GPU Ratio)
    if (argc < 4) {
        if (rank == 0) {
            cout << "Uso: mpirun -np X ./kmeans_openmp <INPUT> <K> <OUT-DIR> [MODO] [CHUNKS] [GPU_RATIO]" << endl;
            cout << "Modos: 0 (CPU) | 1 (GPU) | 2 (Hibrido Pipeline) | 3 (Hibrido Carga Manual)" << endl;
            cout << "Chunks: 0 para automatico, ou valor > 0 para manual" << endl;
            cout << "GPU_RATIO: (Apenas Modo 3) Valor entre 0.0 e 1.0 indicando a % de chunks na GPU." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    int mode = (argc >= 5) ? stoi(argv[4]) : 0;
    int requested_chunks = (argc >= 6) ? stoi(argv[5]) : 0;
    double gpu_ratio = (argc >= 7) ? stod(argv[6]) : 0.5; // Default: 50% GPU
    
    const int nIters = 100; 

    // Ponteiros de função para os modos 0, 1 e 2
    typedef void (*assign_fn)(double*, double*, int*, int, int, int);
    typedef void (*calculate_fn)(double*, int*, double*, int*, int, int, int);
    typedef void (*update_fn)(double*, int*, double*, int, int); // <-- NOVO PONTEIRO
    
    assign_fn assign_points = assign_point_to_cluster_cpu;
    calculate_fn calc_sums = calculate_partial_sums_cpu;
    update_fn update_cents = update_centroids_cpu;       // <-- DEFAULT CPU

    #ifdef USE_GPU
        if (mode == 1) { // FULL GPU
            assign_points = assign_point_to_cluster_gpu;
            calc_sums = calculate_partial_sums_gpu;
            update_cents = update_centroids_gpu;         // <-- MODO GPU USA O NOVO KERNEL
            if (rank == 0) cout << ">> MODO 1: FULL GPU" << endl;
        } 
        else if (mode == 2) { // HÍBRIDO ESTÁTICO (Tudo ou nada na função)
            assign_points = assign_point_to_cluster_gpu; 
            calc_sums = calculate_partial_sums_cpu;      
            update_cents = update_centroids_cpu;         // CPU para evitar overhead na fase de update
            if (rank == 0) cout << ">> MODO 2: HIBRIDO PIPELINE (Assign: GPU | Sums/Update: CPU)" << endl;
        } 
        else if (mode == 3) {
            update_cents = update_centroids_cpu;         // CPU para evitar overhead na fase de update
            if (rank == 0) cout << ">> MODO 3: BALANCEAMENTO MANUAL DE CARGA (" 
                                << (gpu_ratio * 100) << "% GPU | " 
                                << ((1.0 - gpu_ratio) * 100) << "% CPU)" << endl;
        }
        else { // FULL CPU
            if (rank == 0) cout << ">> MODO 0: FULL CPU (OpenMP)" << endl;
        }
    #else
        if (rank == 0 && mode != 0) {
            cout << ">> AVISO: Binário compilado sem suporte a GPU (USE_GPU off). Forçando FULL CPU." << endl;
            mode = 0;
        }
    #endif

    int N = 0, dimensions = 0;
    double *global_points = nullptr;
    int *global_labels = nullptr;
    double *global_centroids = nullptr;

    // Leitura e Inicialização no Mestre (Rank 0)
    if (rank == 0) {
        vector<Point> all_points;
        if (!read_points_from_file(filename, all_points, N, dimensions)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        global_points = new double[N * dimensions];
        global_labels = new int[N];
        global_centroids = new double[K * dimensions];

        for (int i = 0; i < N; i++) {
            for (int d = 0; d < dimensions; d++) {
                global_points[i * dimensions + d] = all_points[i].getVal(d);
            }
            global_labels[i] = 0;
        }
        
        // Inicialização randômica dos centroides (com semente fixa)
        srand(42); 
        std::vector<int> chosen_indices;
        while ((int)chosen_indices.size() < K) {
            int r = rand() % N; 
            if (std::find(chosen_indices.begin(), chosen_indices.end(), r) == chosen_indices.end()) {
                chosen_indices.push_back(r);
            }
        }

        for (int i = 0; i < K; ++i) {
            int point_idx = chosen_indices[i];
            for (int d = 0; d < dimensions; d++) {
                global_centroids[i * dimensions + d] = all_points[point_idx].getVal(d);
            }
        }
    }

    // Broadcast das dimensões e K para todos os nodos
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        global_centroids = new double[K * dimensions];
    }
    MPI_Bcast(global_centroids, K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Preparação para Distribuição dos Pontos (Scatterv)
    int base = N / size;
    int rem = N % size;
    int *sendCountsPts = new int[size];
    int *displsPts = new int[size];
    int *sendCountsLbls = new int[size];
    int *displsLbls = new int[size];

    int offsetLbl = 0, offsetPt = 0;
    for (int i = 0; i < size; i++) {
        int count = (i < rem) ? base + 1 : base;
        sendCountsLbls[i] = count;
        displsLbls[i] = offsetLbl;
        offsetLbl += count;
        sendCountsPts[i] = count * dimensions;
        displsPts[i] = offsetPt;
        offsetPt += count * dimensions;
    }

    int local_n = sendCountsLbls[rank];
    double *local_points = new double[local_n * dimensions];
    int *local_labels = new int[local_n];

    MPI_Scatterv(global_points, sendCountsPts, displsPts, MPI_DOUBLE,
                 local_points, sendCountsPts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- DEFINIÇÃO DOS CHUNKS ---
    int num_chunks;
    if (requested_chunks > 0) {
        num_chunks = requested_chunks;
    } else {
        num_chunks = omp_get_max_threads(); 
        if (num_chunks > local_n && local_n > 0) {
            num_chunks = local_n;
        }
        if (num_chunks <= 0) num_chunks = 1;
    }
    int chunk_size = (local_n + num_chunks - 1) / num_chunks;

    // Calcula quantos chunks vão para a GPU no Modo 3
    int gpu_chunks_target = (int)(num_chunks * gpu_ratio);

    if (rank == 0) {
        cout << ">> Processando " << local_n << " pontos por Rank em " 
             << num_chunks << " chunk(s) " 
             << (requested_chunks > 0 ? "(Manual)" : "(Automatico)") << "." << endl;
        if (mode == 3) {
            cout << ">> Dividindo: " << gpu_chunks_target << " chunks p/ GPU e " 
                 << (num_chunks - gpu_chunks_target) << " chunks p/ CPU." << endl;
        }
    }
    // ----------------------------

    double *local_sums = new double[K * dimensions];
    int *local_counts = new int[K];
    double *global_sums = new double[K * dimensions];
    int *global_counts = new int[K];

    // Barreira antes de começar a contar o tempo
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = high_resolution_clock::now();

    for (int iter = 0; iter < nIters; iter++) {
        memset(local_sums, 0, K * dimensions * sizeof(double));
        memset(local_counts, 0, K * sizeof(int));

        for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
            int offset = chunk_id * chunk_size;
            
            int this_chunk = min(chunk_size, local_n - offset);
            if (this_chunk <= 0) break; // Trava de segurança

            if (mode == 3) {
                // --- MODO 3: Balanceamento Espacial Manual ---
                #ifdef USE_GPU
                if (chunk_id < gpu_chunks_target) {
                    assign_point_to_cluster_gpu(&local_points[offset * dimensions], global_centroids, 
                                                &local_labels[offset], this_chunk, K, dimensions);
                    calculate_partial_sums_gpu(&local_points[offset * dimensions], &local_labels[offset], 
                                               local_sums, local_counts, this_chunk, K, dimensions);
                } else 
                #endif
                {
                    assign_point_to_cluster_cpu(&local_points[offset * dimensions], global_centroids, 
                                                &local_labels[offset], this_chunk, K, dimensions);
                    calculate_partial_sums_cpu(&local_points[offset * dimensions], &local_labels[offset], 
                                               local_sums, local_counts, this_chunk, K, dimensions);
                }
            } else {
                // --- MODOS 0, 1 e 2: Comportamento Tradicional via ponteiro de função ---
                assign_points(&local_points[offset * dimensions], global_centroids, 
                              &local_labels[offset], this_chunk, K, dimensions);
                              
                calc_sums(&local_points[offset * dimensions], &local_labels[offset], 
                          local_sums, local_counts, this_chunk, K, dimensions);
            }
        }
        // ----------------------------------------

        // Sincronização global síncrona dos centroides parciais
        MPI_Allreduce(local_sums, global_sums, K * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // =====================================================================
        // ATUALIZAÇÃO DOS CENTROIDES GLOBAIS (VIA KERNEL ABSTRAÍDO)
        // =====================================================================
        update_cents(global_sums, global_counts, global_centroids, K, dimensions);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = high_resolution_clock::now();

    // Reúne os labels finais no rank 0
    MPI_Gatherv(local_labels, local_n, MPI_INT, 
                global_labels, sendCountsLbls, displsLbls, MPI_INT, 0, MPI_COMM_WORLD);

    // Impressão dos Resultados e Validação GPU
    if (rank == 0) {
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        cout << "\nExecution time: " << duration.count() << " ms" << endl;

#ifdef USE_GPU
        printf("\n========================================\n");
        printf("[VERIFICACAO DE OFFLOAD - OPENMP]\n");
        // Acessa as variáveis externas definidas no arquivo das funções GPU
        extern int cuda_assign_calls;
        extern int cuda_calculate_calls;
        extern int cuda_update_calls; // <-- ADICIONADO PARA O NOVO KERNEL
        printf("Chamadas na GPU (Assign): %d\n", cuda_assign_calls);
        printf("Chamadas na GPU (Calculate): %d\n", cuda_calculate_calls);
        printf("Chamadas na GPU (Update): %d\n", cuda_update_calls); // <-- EXIBE O CONTADOR
        printf("========================================\n");
#endif
    }

    // Limpeza de memória
    delete[] local_points; delete[] local_labels;
    delete[] local_sums; delete[] local_counts;
    delete[] global_sums; delete[] global_counts;
    delete[] global_centroids;
    delete[] sendCountsPts; delete[] displsPts;
    delete[] sendCountsLbls; delete[] displsLbls;
    if (rank == 0) { delete[] global_points; delete[] global_labels; }

    MPI_Finalize();
    return 0;
}