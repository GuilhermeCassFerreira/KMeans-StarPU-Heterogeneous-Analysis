#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "../../include/kmeans_types.h"
#include "kmeans_omp_mpi.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

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

    if (argc < 4) {
        if (rank == 0) {
            cout << "Uso: mpirun -np X ./kmeans_openmp <INPUT> <K> <OUT-DIR> [MODO] [CHUNKS] [GPU_RATIO] [SEED] [NITERS]" << endl;
            cout << "Modos: 0 (CPU) | 1 (GPU) | 2 (Hibrido)" << endl;
            cout << "Chunks: 0 para automatico, ou valor > 0 para manual" << endl;
            cout << "GPU_RATIO: (Modo 2) % de chunks na GPU (0.0 a 1.0)" << endl;
            cout << "SEED: Semente para geração aleatória (inteiro)" << endl;
            cout << "NITERS: Número de iterações (inteiro)" << endl;
        }
        MPI_Finalize(); return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    string output_dir = argv[3];
    int mode = (argc >= 5) ? stoi(argv[4]) : 0;
    int requested_chunks = (argc >= 6) ? stoi(argv[5]) : 0;
    double gpu_ratio = (argc >= 7) ? stod(argv[6]) : 0.5;
    unsigned int seed = (argc >= 8) ? (unsigned int)stoul(argv[7]) : 42;
    int nIters = (argc >= 9) ? stoi(argv[8]) : 100;

    assign_fn assign_points = assign_point_to_cluster_cpu;
    calculate_fn calc_sums = calculate_partial_sums_cpu;
    update_fn update_cents = update_centroids_cpu;       

    #ifdef USE_GPU
        if (mode == 1) { 
            assign_points = assign_point_to_cluster_gpu;
            calc_sums = calculate_partial_sums_gpu;
            update_cents = update_centroids_gpu;       
            if (rank == 0) cout << ">> MODO 1: FULL GPU (Com Pinned Memory)" << endl;
        } 
        else if (mode == 2) {
            update_cents = update_centroids_gpu;        
            if (rank == 0) cout << ">> MODO 2: HIBRIDO - BALANCEAMENTO MANUAL (" 
                                << (gpu_ratio * 100) << "% GPU | " 
                                << ((1.0 - gpu_ratio) * 100) << "% CPU) (Com Pinned Memory)" << endl;
        }
        else { 
            if (rank == 0) cout << ">> MODO 0: FULL CPU (OpenMP)" << endl;
        }
    #else
        if (rank == 0 && mode != 0) {
            cout << ">> AVISO: Binário compilado sem suporte a GPU (USE_GPU off). Forçando FULL CPU." << endl;
            mode = 0;
        }
    #endif

    int N = 0, dimensions = 0;
    double *global_points = nullptr, *global_centroids = nullptr;
    int *global_labels = nullptr;

    if (rank == 0) {
        vector<Point> all_points;
        if (!read_points_from_file(filename, all_points, N, dimensions)) MPI_Abort(MPI_COMM_WORLD, 1);
        global_points = new double[N * dimensions];
        global_labels = new int[N];
        global_centroids = new double[K * dimensions];

        for (int i = 0; i < N; i++) {
            for (int d = 0; d < dimensions; d++) global_points[i * dimensions + d] = all_points[i].getVal(d);
            global_labels[i] = 0;
        }
        srand(seed);
        cout << ">> Rank 0 inicializando centroides com SEED: " << seed << endl;

        // Inicialização sem repetição — igual SEQ e StarPU
        vector<int> chosen_indices;
        while ((int)chosen_indices.size() < K) {
            int r = rand() % N;
            if (find(chosen_indices.begin(), chosen_indices.end(), r) == chosen_indices.end()) {
                chosen_indices.push_back(r);
            }
        }
        for (int i = 0; i < K; ++i) {
            int p_idx = chosen_indices[i];
            for (int d = 0; d < dimensions; d++) global_centroids[i * dimensions + d] = global_points[p_idx * dimensions + d];
        }
    } 

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) global_centroids = new double[K * dimensions];
    MPI_Bcast(global_centroids, K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    #ifdef USE_GPU
    if (mode == 1 || mode == 2) {
        cudaHostRegister(global_centroids, K * dimensions * sizeof(double), cudaHostRegisterDefault);
    }
    #endif

    int base = N / size, rem = N % size;
    int *sendCountsPts = new int[size], *displsPts = new int[size];
    int *sendCountsLbls = new int[size], *displsLbls = new int[size];
    int offL = 0, offP = 0;
    for (int i = 0; i < size; i++) {
        int c = (i < rem) ? base + 1 : base;
        sendCountsLbls[i] = c; displsLbls[i] = offL; offL += c;
        sendCountsPts[i] = c * dimensions; displsPts[i] = offP; offP += c * dimensions;
    }

    int local_n = sendCountsLbls[rank];
    double *local_points = new double[local_n * dimensions];
    int *local_labels = new int[local_n];

    #ifdef USE_GPU
    if (mode == 1 || mode == 2) {
        cudaHostRegister(local_points, local_n * dimensions * sizeof(double), cudaHostRegisterDefault);
        cudaHostRegister(local_labels, local_n * sizeof(int), cudaHostRegisterDefault);
    }
    #endif

    MPI_Scatterv(global_points, sendCountsPts, displsPts, MPI_DOUBLE, local_points, sendCountsPts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int num_chunks = requested_chunks;
    int chunk_size = (local_n + num_chunks - 1) / num_chunks;
    int gpu_chunks_target = (int)(num_chunks * gpu_ratio);

    if (rank == 0) {
        cout << ">> Processando " << N << " pontos totais em " << size << " Rank(s)." << endl;
        cout << ">> Cada Rank processa aproximadamente " << local_n << " pontos em " << num_chunks << " chunk(s)." << endl;
        if (mode == 2) {
            cout << ">> Divisão hibrida por Rank: " << gpu_chunks_target << " chunks na GPU e " 
                 << (num_chunks - gpu_chunks_target) << " na CPU." << endl;
        }
    }

    double *local_sums = new double[K * dimensions], *global_sums = new double[K * dimensions];
    int *local_counts = new int[K], *global_counts = new int[K];

    #ifdef USE_GPU
    if (mode == 1 || mode == 2) {
        cudaHostRegister(local_sums, K * dimensions * sizeof(double), cudaHostRegisterDefault);
        cudaHostRegister(local_counts, K * sizeof(int), cudaHostRegisterDefault);
    }
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = high_resolution_clock::now();
    auto t_converge = t_start;
    int iter_converged = -1;

    #ifdef USE_GPU
        #pragma omp target enter data map(to: local_points[0:local_n*dimensions]) \
                                   map(to: local_labels[0:local_n])
    #endif

    for (int iter = 0; iter < nIters; iter++) {
        memset(local_sums, 0, K * dimensions * sizeof(double));
        memset(local_counts, 0, K * sizeof(int));
        int local_changes = 0;

        for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
            int offset = chunk_id * chunk_size;
            int this_chunk = min(chunk_size, local_n - offset);
            if (this_chunk <= 0) break; 

            if (mode == 2) {
                #ifdef USE_GPU
                if (chunk_id < gpu_chunks_target) {
                    local_changes += assign_point_to_cluster_gpu(&local_points[offset * dimensions], global_centroids, &local_labels[offset], this_chunk, K, dimensions);
                    calculate_partial_sums_gpu(&local_points[offset * dimensions], &local_labels[offset], local_sums, local_counts, this_chunk, K, dimensions);
                } else 
                #endif
                {
                    local_changes += assign_point_to_cluster_cpu(&local_points[offset * dimensions], global_centroids, &local_labels[offset], this_chunk, K, dimensions);
                    calculate_partial_sums_cpu(&local_points[offset * dimensions], &local_labels[offset], local_sums, local_counts, this_chunk, K, dimensions);
                }
            } else {
                local_changes += assign_points(&local_points[offset * dimensions], global_centroids, &local_labels[offset], this_chunk, K, dimensions);
                calc_sums(&local_points[offset * dimensions], &local_labels[offset], local_sums, local_counts, this_chunk, K, dimensions);
            }
        }

        int global_changes = 0;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_changes == 0) {
            iter_converged = iter + 1;
            t_converge = high_resolution_clock::now();
            if (rank == 0) cout << ">> Convergiu na iteracao " << iter + 1 << endl;
            break;
        }
        if (rank == 0) cout << ">> Iteracao " << iter + 1 << " teve " << global_changes << " mudancas." << endl;

        MPI_Allreduce(local_sums, global_sums, K * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        update_cents(global_sums, global_counts, global_centroids, K, dimensions);

        if (iter == nIters - 1) {
            iter_converged = nIters;       // limite atingido sem convergir
            t_converge = high_resolution_clock::now();
        }
    }

    #ifdef USE_GPU
        #pragma omp target exit data map(from: local_labels[0:local_n])
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = high_resolution_clock::now();

    // ----- Coleta dos labels finais no rank 0 e gravação dos arquivos -----
    // Mesmo formato de saída que SEQ e StarPU (K-points.txt, K-clusters.txt)
    MPI_Gatherv(local_labels, local_n, MPI_INT,
                global_labels, sendCountsLbls, displsLbls, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sse = 0.0;
        for (int i = 0; i < N; i++) {
            int c = global_labels[i];
            for (int d = 0; d < dimensions; d++) {
                double diff = global_points[i * dimensions + d] - global_centroids[c * dimensions + d];
                sse += diff * diff;
            }
        }

        double t_total_ms    = duration<double, milli>(t_end      - t_start).count();
        double t_converge_ms = duration<double, milli>(t_converge - t_start).count();

        cout << "\n========================================" << endl;
        cout << "METRICAS FINAIS (OpenMP/MPI)" << endl;
        cout << "========================================" << endl;
        cout << fixed << setprecision(4);
        cout << "SSE (Soma dos Erros Quadraticos):  " << sse << endl;
        cout << "Iteracoes ate convergir:           " << iter_converged << " / " << nIters << endl;
        cout << "Tempo ate convergencia:            " << t_converge_ms << " ms" << endl;
        cout << "Tempo total (com I/O final):       " << t_total_ms    << " ms" << endl;
        cout << "Nos MPI utilizados:                " << size << endl;
        cout << "========================================" << endl;

#ifdef USE_GPU
        printf("\n========================================\n");
        printf("[VERIFICACAO DE OFFLOAD - OPENMP]\n");
        printf("Chamadas na GPU (Assign): %d\n", cuda_assign_calls);
        printf("Chamadas na GPU (Calculate): %d\n", cuda_calculate_calls);
        printf("Chamadas na GPU (Update): %d\n", cuda_update_calls);
        printf("========================================\n");
#endif

        string cmd = "mkdir -p " + output_dir;
        if (system(cmd.c_str()) != 0) {
            cerr << "[AVISO] Falha ao criar diretório: " << output_dir << endl;
        }

        ofstream pointsFile(output_dir + "/" + to_string(K) + "-points.txt");
        for (int i = 0; i < N; i++) {
            pointsFile << global_labels[i] << "\n";
        }
        pointsFile.close();

        ofstream clustersFile(output_dir + "/" + to_string(K) + "-clusters.txt");
        clustersFile << fixed << setprecision(6);
        for (int k = 0; k < K; k++) {
            for (int d = 0; d < dimensions; d++) {
                clustersFile << global_centroids[k * dimensions + d] << " ";
            }
            clustersFile << "\n";
        }
        clustersFile.close();

        cout << "[INFO] Arquivos salvos em: " << output_dir << endl;
    }

    #ifdef USE_GPU
    if (mode == 1 || mode == 2) {
        cudaHostUnregister(local_sums);
        cudaHostUnregister(local_counts);
        cudaHostUnregister(local_points);
        cudaHostUnregister(local_labels);
        cudaHostUnregister(global_centroids);
    }
    #endif

    delete[] local_points; delete[] local_labels; delete[] local_sums; delete[] local_counts;
    delete[] global_sums; delete[] global_counts; delete[] global_centroids;
    if (rank == 0) { delete[] global_points; delete[] global_labels; }
    MPI_Finalize();
    return 0;
}