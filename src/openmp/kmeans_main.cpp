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

    if (argc < 4) {
        if (rank == 0) {
            cout << "Uso: mpirun -np X ./kmeans_openmp <INPUT> <K> <OUT-DIR> [MODO] [CHUNKS] [GPU_RATIO] [SEED]" << endl;
            cout << "Modos: 0 (CPU) | 1 (GPU) | 2 (Hibrido)" << endl;
            cout << "Chunks: 0 para automatico, ou valor > 0 para manual" << endl;
            cout << "GPU_RATIO: (Modo 2) % de chunks na GPU (0.0 a 1.0)" << endl;
            cout << "SEED: Semente para geração aleatória (inteiro)" << endl;
        }
        MPI_Finalize(); return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    int mode = (argc >= 5) ? stoi(argv[4]) : 0;
    int requested_chunks = (argc >= 6) ? stoi(argv[5]) : 0;
    double gpu_ratio = (argc >= 7) ? stod(argv[6]) : 0.5;
    unsigned int seed = (argc >= 8) ? (unsigned int)stoul(argv[7]) : 42;
    
    const int nIters = 1; 

    // Inicialização usando os tipos do Header
    assign_fn assign_points = assign_point_to_cluster_cpu;
    calculate_fn calc_sums = calculate_partial_sums_cpu;
    update_fn update_cents = update_centroids_cpu;       

    #ifdef USE_GPU
        if (mode == 1) { 
            assign_points = assign_point_to_cluster_gpu;
            calc_sums = calculate_partial_sums_gpu;
            update_cents = update_centroids_gpu;       
            if (rank == 0) cout << ">> MODO 1: FULL GPU" << endl;
        } 
        else if (mode == 2) {
            update_cents = update_centroids_gpu;        
            if (rank == 0) cout << ">> MODO 2: HIBRIDO - BALANCEAMENTO MANUAL (" 
                                << (gpu_ratio * 100) << "% GPU | " 
                                << ((1.0 - gpu_ratio) * 100) << "% CPU)" << endl;
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
        if (rank == 0) cout << ">> Rank 0 inicializando centroides com SEED: " << seed << endl;
        for (int i = 0; i < K; ++i) {
            int p_idx = rand() % N;
            for (int d = 0; d < dimensions; d++) global_centroids[i * dimensions + d] = global_points[p_idx * dimensions + d];
        }
    } 

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) global_centroids = new double[K * dimensions];
    MPI_Bcast(global_centroids, K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    MPI_Scatterv(global_points, sendCountsPts, displsPts, MPI_DOUBLE, local_points, sendCountsPts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int num_chunks = (requested_chunks > 0) ? requested_chunks : max(1, (int)omp_get_max_threads());
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_t = high_resolution_clock::now();
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
            if (rank == 0) cout << ">> Convergiu na iteracao " << iter + 1 << endl;
            break;
        }
        cout << ">> Iteracao " << iter + 1 << " teve " << global_changes << " mudancas." << endl;

        MPI_Allreduce(local_sums, global_sums, K * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        update_cents(global_sums, global_counts, global_centroids, K, dimensions);
    }

    #ifdef USE_GPU
        #pragma omp target exit data map(from: local_labels[0:local_n])
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        auto dur = duration_cast<milliseconds>(high_resolution_clock::now() - start_t);
        cout << "\nExecution time: " << dur.count() << " ms" << endl;

#ifdef USE_GPU
        printf("\n========================================\n");
        printf("[VERIFICACAO DE OFFLOAD - OPENMP]\n");
        printf("Chamadas na GPU (Assign): %d\n", cuda_assign_calls);
        printf("Chamadas na GPU (Calculate): %d\n", cuda_calculate_calls);
        printf("Chamadas na GPU (Update): %d\n", cuda_update_calls);
        printf("========================================\n");
#endif
    }

    delete[] local_points; delete[] local_labels; delete[] local_sums; delete[] local_counts;
    delete[] global_sums; delete[] global_counts; delete[] global_centroids;
    if (rank == 0) { delete[] global_points; delete[] global_labels; }
    MPI_Finalize();
    return 0;
}