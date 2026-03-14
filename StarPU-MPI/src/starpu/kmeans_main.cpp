#include "kmeans_runtime.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <iomanip>

using namespace std;
using namespace chrono;

int main(int argc, char **argv) {
    int rank, size; // pega o rank e o tamanho do comunicador 
    int mpi_provided;
    auto start = high_resolution_clock::now();

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Parsing de argumentos ----
    vector<string> args;
    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }

    if (args.size() < 3 || args.size() > 4) {
        if (rank == 0)
            cout << "Error: command-line argument count mismatch.\n"
                 << " ./kmeans_starpu <INPUT> <K> <OUT-DIR> [CHUNK_SIZE]" << endl;
        MPI_Finalize();
        return 1;
    }

    string filename = args[0];
    int K = stoi(args[1]);
    string output_dir = args[2];
    int chunk_size = (args.size() == 4) ? stoi(args[3]) : -1;

    // ---- Leitura dos pontos (apenas no nodo 0) ----
    vector<Point> all_points;
    int N = 0;
    int dimensions = 0;

    if (rank == 0) {
        if (!read_points_from_file(filename, all_points, N, dimensions)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N < K) {
        if (rank == 0) cout << "Error: Number of clusters greater than number of points." << endl;
        MPI_Finalize();
        return 1;
    }

    int iters = 100;

    // ---- Inicialização do StarPU-MPI ----
    int ret = starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, NULL);
    if (ret != 0) {
        if (rank == 0) cerr << "Error: Failed to initialize StarPU-MPI." << endl;
        MPI_Finalize();
        return 1;
    }

    // ---- Auto-configuração do chunk_size ----
    unsigned local_cpus = starpu_cpu_worker_get_count();
    unsigned local_gpus = 0;
    #ifdef STARPU_USE_CUDA
        local_gpus = starpu_cuda_worker_get_count();
    #endif
    cout <<  "Total de CPUs locais: " << local_cpus << " | Total de GPUs locais: " << local_gpus << endl;

    unsigned global_cpus = 0;
    unsigned global_gpus = 0;

    MPI_Reduce(&local_cpus, &global_cpus, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gpus, &global_gpus, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 && chunk_size == -1) {
        unsigned gpu_weight = 15;
        unsigned virtual_power = global_cpus + (global_gpus * gpu_weight);

        double multiplicity = (global_gpus > 0) ? 8.0 : 4.0;
        double density_ratio = (double)N / (virtual_power > 0 ? virtual_power : 1);

        if (density_ratio < 1000.0) multiplicity *= 2.0;
        else if (density_ratio > 100000.0) multiplicity /= 2.0;

        int desired_num_chunks = max(1, (int)((double)virtual_power * multiplicity));
        chunk_size = max(1, (int)((N + desired_num_chunks - 1) / desired_num_chunks));

        cout << "[AUTO-CONFIG] Arquitetura: " << global_cpus << " CPUs e " << global_gpus << " GPUs globais." << endl;
        cout << "[AUTO-CONFIG] Poder Virtual: " << virtual_power << " | Multiplicidade: " << multiplicity << "x" << endl;
        cout << "[AUTO-CONFIG] Chunks: " << desired_num_chunks << " | Chunk Size: " << chunk_size << endl;
    }

    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool use_heterogeneous_chunks_val = false;

    KMeans kmeans(K, iters, output_dir, chunk_size, use_heterogeneous_chunks_val, rank, size, dimensions);
    kmeans.run(all_points, N);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    if (rank == 0) cout << "\nExecution time: " << duration.count() << " ms" << endl;

    print_starpu_worker_usage(rank);
    print_kernel_usage_metrics(rank);
    print_node_usage_metrics(rank, size);

    // ---- Finalização ----
    starpu_mpi_shutdown();

    if (rank == 0) {
        const char* sched = getenv("STARPU_SCHED");
        if (sched) cout << "Escalonador StarPU ativo: " << sched << endl;
    }

    MPI_Finalize();
    return 0;
}