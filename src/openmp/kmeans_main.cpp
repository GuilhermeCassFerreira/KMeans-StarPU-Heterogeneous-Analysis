#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm> // Necessário para o std::find
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

    if (argc < 4) {
        if (rank == 0) {
            cout << "Uso: mpirun -np X ./kmeans_openmp <INPUT> <K> <OUT-DIR> [MODO]" << endl;
            cout << "Modos: 0 (CPU) | 1 (GPU) | 2 (Hibrido)" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    string output_dir = argv[3];
    int mode = (argc >= 5) ? stoi(argv[4]) : 0;
    
    // Bate de frente com os 100 do seu StarPU
    const int nIters = 100; 

    // Seleção de hardware (Contrato de ponteiros)
    assign_fn assign_points = assign_point_to_cluster_cpu;
    calculate_fn calc_sums = calculate_partial_sums_cpu;

    if (mode == 1) {
        assign_points = assign_point_to_cluster_gpu;
        calc_sums = calculate_partial_sums_gpu;
        if (rank == 0) cout << ">> Modo: 100% GPU (OpenMP Target)" << endl;
    } else if (mode == 2) {
        assign_points = assign_point_to_cluster_gpu; 
        calc_sums = calculate_partial_sums_cpu;      
        if (rank == 0) cout << ">> Modo: HIBRIDO (Assign: GPU | Sums: CPU)" << endl;
    } else {
        if (rank == 0) cout << ">> Modo: 100% CPU (OpenMP)" << endl;
    }

    int N = 0, dimensions = 0;
    double *global_points = nullptr;
    int *global_labels = nullptr;
    double *global_centroids = nullptr;

    // Leitura e Inicialização no Master (Igual ao seu StarPU)
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
        
        // --- INÍCIO DA INICIALIZAÇÃO ALEATÓRIA (SEED 42) ---
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
        cout << "[INFO] Centroides iniciais escolhidos aleatoriamente (Seed: 42)" << endl;
        // --- FIM DA INICIALIZAÇÃO ALEATÓRIA ---
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        global_centroids = new double[K * dimensions];
    }
    MPI_Bcast(global_centroids, K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divisão de Carga Estática
    int base = N / size; // minimo de pontos que cada nodo vai receber
    int rem = N % size;//resto caso seja N = 10 e size = 3, rem = 1 
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

    double *local_sums = new double[K * dimensions];
    int *local_counts = new int[K];
    double *global_sums = new double[K * dimensions];
    int *global_counts = new int[K];

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = high_resolution_clock::now();

    // Loop de Treinamento
    for (int iter = 0; iter < nIters; iter++) {
        memset(local_sums, 0, K * dimensions * sizeof(double));
        memset(local_counts, 0, K * sizeof(int));

        assign_points(local_points, global_centroids, local_labels, local_n, K, dimensions);
        calc_sums(local_points, local_labels, local_sums, local_counts, local_n, K, dimensions);

        // Equivalente ao seu reduceCentroidsAcrossNodes()
        MPI_Allreduce(local_sums, global_sums, K * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int c = 0; c < K; c++) {
            if (global_counts[c] > 0) {
                for (int d = 0; d < dimensions; d++) {
                    global_centroids[c * dimensions + d] = global_sums[c * dimensions + d] / global_counts[c];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = high_resolution_clock::now();

    // Coleta final dos labels
    MPI_Gatherv(local_labels, local_n, MPI_INT, 
                global_labels, sendCountsLbls, displsLbls, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        cout << "\nExecution time: " << duration.count() << " ms" << endl;

        // --- INÍCIO DA VERIFICAÇÃO DE OFFLOAD ---
        extern int cuda_assign_calls;
        extern int cuda_calculate_calls;
        
        printf("\n========================================\n");
        printf("[VERIFICACAO DE OFFLOAD - OPENMP]\n");
        printf("Chamadas na GPU (Assign): %d\n", cuda_assign_calls);
        printf("Chamadas na GPU (Calculate): %d\n", cuda_calculate_calls);
        printf("========================================\n");
        // --- FIM DA VERIFICAÇÃO DE OFFLOAD ---
    }

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