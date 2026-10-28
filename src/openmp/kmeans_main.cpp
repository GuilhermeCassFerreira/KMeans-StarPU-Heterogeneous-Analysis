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
#include "../../include/metrics.h"
#include "kmeans_omp_mpi.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

using namespace std;
using namespace chrono;

extern bool read_points_flat(const string& filename, vector<double>& values, int& N, int& dimensions);

#ifdef USE_GPU
// Divide [begin, end) entre nworkers; devolve a faixa [b, e) do worker w
static void split_range(int begin, int end, int w, int nworkers, int &b, int &e) {
    int len = end - begin, base = len / nworkers, rem = len % nworkers;
    b = begin + w * base + min(w, rem);
    e = b + base + (w < rem ? 1 : 0);
}

// Modo 2 (híbrido concorrente): a GPU processa os pontos [0, gpu_n) enquanto as
// threads de CPU processam [gpu_n, n). A thread 0 só controla a GPU; as demais
// dividem a parte da CPU. Dentro desta região, o "parallel for" dos kernels de
// CPU roda com 1 thread (paralelismo aninhado desativado).
static int hybrid_assign_and_sums(double *points, double *centroids, int *labels, int n, int gpu_n,
                                  int K, int dimensions, int cpu_threads,
                                  double *gpu_sums, int *gpu_counts,
                                  double *cpu_sums, int *cpu_counts) {
    int changes = 0;
    #pragma omp parallel num_threads(1 + cpu_threads) reduction(+:changes)
    {
        int tid = omp_get_thread_num();
        int workers = omp_get_num_threads() - 1;

        if (tid == 0 && gpu_n > 0) {
            changes += assign_point_to_cluster_gpu(points, centroids, labels, gpu_n, K, dimensions);
            calculate_partial_sums_gpu(points, labels, gpu_sums, gpu_counts, gpu_n, K, dimensions);
        }

        // Sem threads extras, a thread 0 também faz a parte da CPU
        int w = (workers == 0) ? 0 : tid - 1;
        if (workers == 0 || tid > 0) {
            int b, e;
            split_range(gpu_n, n, w, max(workers, 1), b, e);
            if (e > b) {
                changes += assign_point_to_cluster_cpu(&points[(size_t)b * dimensions], centroids,
                                                       &labels[b], e - b, K, dimensions);
                calculate_partial_sums_cpu(&points[(size_t)b * dimensions], &labels[b],
                                           &cpu_sums[(size_t)w * K * dimensions], &cpu_counts[(size_t)w * K],
                                           e - b, K, dimensions);
            }
        }
    }
    return changes;
}

// Modo 2: a GPU atualiza os centroides [0, gpu_k) e a CPU atualiza [gpu_k, K) ao mesmo tempo
static void hybrid_update_centroids(double *global_sums, int *global_counts, double *centroids,
                                    int K, int gpu_k, int dimensions) {
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        if (tid == 0 && gpu_k > 0)
            update_centroids_gpu(global_sums, global_counts, centroids, gpu_k, dimensions);
        if ((tid == 1 || nth == 1) && gpu_k < K)
            update_centroids_cpu(&global_sums[(size_t)gpu_k * dimensions], &global_counts[gpu_k],
                                 &centroids[(size_t)gpu_k * dimensions], K - gpu_k, dimensions);
    }
}
#endif

int main(int argc, char **argv) {
    auto t_prog_start = high_resolution_clock::now();
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
    vector<double> all_values;   // pontos lidos pelo rank 0 (texto ou binário)

    if (rank == 0) {
        if (!read_points_flat(filename, all_values, N, dimensions)) MPI_Abort(MPI_COMM_WORLD, 1);
        global_points = all_values.data();
        global_labels = new int[N]();
        global_centroids = new double[K * dimensions];
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

    // Modo 2: divisão por pontos e por centroides segundo o gpu_ratio (os chunks não se aplicam)
    gpu_ratio = min(max(gpu_ratio, 0.0), 1.0);
    int gpu_n = (int)(local_n * gpu_ratio);
    int gpu_k = (int)(K * gpu_ratio);
    int cpu_threads = max(omp_get_max_threads() - 1, 1);

    if (rank == 0) {
        cout << ">> Processando " << N << " pontos totais em " << size << " Rank(s)." << endl;
        if (mode == 2) {
            cout << ">> Divisão hibrida por Rank (concorrente): " << gpu_n << " pontos na GPU e "
                 << (local_n - gpu_n) << " na CPU | centroides: " << gpu_k << " na GPU e " << (K - gpu_k)
                 << " na CPU | threads: 1 (GPU) + " << cpu_threads << " (CPU)" << endl;
        } else {
            cout << ">> Cada Rank processa aproximadamente " << local_n << " pontos em " << num_chunks << " chunk(s)." << endl;
        }
    }

    double *local_sums = new double[K * dimensions], *global_sums = new double[K * dimensions];
    int *local_counts = new int[K], *global_counts = new int[K];

    // Acumuladores separados por dispositivo (modo 2): GPU e cada thread de CPU somam em buffers próprios
    double *gpu_sums = new double[K * dimensions];
    int *gpu_counts = new int[K];
    vector<double> cpu_sums((size_t)cpu_threads * K * dimensions);
    vector<int> cpu_counts((size_t)cpu_threads * K);

    #ifdef USE_GPU
    if (mode == 1 || mode == 2) {
        cudaHostRegister(local_sums, K * dimensions * sizeof(double), cudaHostRegisterDefault);
        cudaHostRegister(local_counts, K * sizeof(int), cudaHostRegisterDefault);
    }
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = high_resolution_clock::now();
    int iter_converged = -1;

    #ifdef USE_GPU
    if (mode == 1) {
        #pragma omp target enter data map(to: local_points[0:local_n*dimensions]) \
                                   map(to: local_labels[0:local_n])
    }
    if (mode == 2) {
        // Só a parte da GPU fica residente; a parte da CPU continua no host
        #pragma omp target enter data map(to: local_points[0:gpu_n*dimensions]) \
                                   map(to: local_labels[0:gpu_n])
        // Garante que o "parallel for" dos kernels de CPU rode com 1 thread dentro da região híbrida
        omp_set_max_active_levels(1);
    }
    #endif

    for (int iter = 0; iter < nIters; iter++) {
        memset(local_sums, 0, K * dimensions * sizeof(double));
        memset(local_counts, 0, K * sizeof(int));
        int local_changes = 0;

        if (mode == 2) {
            #ifdef USE_GPU
            memset(gpu_sums, 0, K * dimensions * sizeof(double));
            memset(gpu_counts, 0, K * sizeof(int));
            fill(cpu_sums.begin(), cpu_sums.end(), 0.0);
            fill(cpu_counts.begin(), cpu_counts.end(), 0);

            local_changes = hybrid_assign_and_sums(local_points, global_centroids, local_labels, local_n, gpu_n,
                                                   K, dimensions, cpu_threads, gpu_sums, gpu_counts,
                                                   cpu_sums.data(), cpu_counts.data());

            for (int i = 0; i < K * dimensions; i++) {
                local_sums[i] = gpu_sums[i];
                for (int t = 0; t < cpu_threads; t++) local_sums[i] += cpu_sums[(size_t)t * K * dimensions + i];
            }
            for (int k = 0; k < K; k++) {
                local_counts[k] = gpu_counts[k];
                for (int t = 0; t < cpu_threads; t++) local_counts[k] += cpu_counts[(size_t)t * K + k];
            }
            #endif
        } else {
            for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
                int offset = chunk_id * chunk_size;
                int this_chunk = min(chunk_size, local_n - offset);
                if (this_chunk <= 0) break;

                local_changes += assign_points(&local_points[offset * dimensions], global_centroids, &local_labels[offset], this_chunk, K, dimensions);
                calc_sums(&local_points[offset * dimensions], &local_labels[offset], local_sums, local_counts, this_chunk, K, dimensions);
            }
        }

        int global_changes = 0;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_changes == 0) {
            iter_converged = iter + 1;
            if (rank == 0) cout << "[OMP] Convergiu (label changes=0) na iteracao " << iter + 1 << endl;
            break;
        }

        MPI_Allreduce(local_sums, global_sums, K * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (mode == 2) {
            #ifdef USE_GPU
            hybrid_update_centroids(global_sums, global_counts, global_centroids, K, gpu_k, dimensions);
            #endif
        } else {
            update_cents(global_sums, global_counts, global_centroids, K, dimensions);
        }

        if (rank == 0) cout << "[OMP] Iteracao " << iter + 1 << " | mudancas: " << global_changes << endl;

        if (iter == nIters - 1) {
            iter_converged = nIters;
        }
    }

    #ifdef USE_GPU
    if (mode == 1) {
        #pragma omp target exit data map(from: local_labels[0:local_n])
    }
    if (mode == 2) {
        #pragma omp target exit data map(from: local_labels[0:gpu_n]) \
                                  map(delete: local_points[0:gpu_n*dimensions])
    }
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = high_resolution_clock::now();

    // ----- Coleta dos labels finais no rank 0 e gravação dos arquivos -----
    // Mesmo formato de saída que SEQ e StarPU (K-points.txt, K-clusters.txt)
    MPI_Gatherv(local_labels, local_n, MPI_INT,
                global_labels, sendCountsLbls, displsLbls, MPI_INT,
                0, MPI_COMM_WORLD);

    double sse = 0.0;
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            int c = global_labels[i] - 1;  // labels sao 1-indexed
            if (c < 0 || c >= K) continue;
            for (int d = 0; d < dimensions; d++) {
                double diff = global_points[i * dimensions + d] - global_centroids[c * dimensions + d];
                sse += diff * diff;
            }
        }

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

    // Todos os ranks preenchem métricas e chamam print (que faz MPI_Gather internamente)
    auto t_prog_end = high_resolution_clock::now();
    double t_loop_ms  = duration<double, milli>(t_end - t_start).count();
    double t_total_ms = duration<double, milli>(t_prog_end - t_prog_start).count();

    OmpMetrics m{};
    m.t_loop_ms       = t_loop_ms;
    m.t_total_ms      = t_total_ms;
    m.iter_converged  = iter_converged;
    m.iter_max        = nIters;
    m.sse             = sse;  // 0.0 nos ranks != 0
    m.mpi_ranks       = size;
    m.omp_threads     = omp_get_max_threads();
    m.mode            = mode;
    m.num_chunks      = num_chunks;
    m.total_assign    = (long)iter_converged * num_chunks;
    m.total_calculate = (long)iter_converged * num_chunks;
    m.total_update    = (long)iter_converged;
#ifdef USE_GPU
    m.gpu_assign    = cuda_assign_calls;
    m.gpu_calculate = cuda_calculate_calls;
    m.gpu_update    = cuda_update_calls;
    if (mode == 2) {
        // Híbrido: contagens em pontos (Assign/Calculate) e centroides (Update);
        // a parte da GPU só conta se o kernel de fato rodou no dispositivo
        m.total_assign    = (long)iter_converged * local_n;
        m.total_calculate = (long)iter_converged * local_n;
        m.total_update    = (long)iter_converged * K;
        m.gpu_assign      = (cuda_assign_calls > 0)    ? (long)iter_converged * gpu_n : 0;
        m.gpu_calculate   = (cuda_calculate_calls > 0) ? (long)iter_converged * gpu_n : 0;
        m.gpu_update      = (cuda_update_calls > 0)    ? (long)iter_converged * gpu_k : 0;
        m.omp_threads     = 1 + cpu_threads;
    }
#endif
    print_omp_metrics(rank, size, m);

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
    delete[] gpu_sums; delete[] gpu_counts;
    if (rank == 0) { delete[] global_labels; }
    MPI_Finalize();
    return 0;
}