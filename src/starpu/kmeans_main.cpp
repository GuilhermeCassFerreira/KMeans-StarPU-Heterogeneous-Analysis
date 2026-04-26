#include "kmeans_runtime.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace chrono;

int main(int argc, char **argv) {
    int rank, size; 
    int mpi_provided;
    auto start = high_resolution_clock::now();

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && mpi_provided < MPI_THREAD_MULTIPLE) {
        cout << "[AVISO] O OpenMPI nao forneceu MPI_THREAD_MULTIPLE." << endl;
        cout << "Nivel fornecido: " << mpi_provided << endl;
        cout << "Isso pode causar gargalos de comunicacao no StarPU." << endl;
    }

    vector<string> args;
    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }

    if (args.size() < 3 || args.size() > 6) {
        if (rank == 0)
            cout << "Uso: ./kmeans_starpu <INPUT> <K> <OUT-DIR> <INTERS> [NUM_CHUNKS] [SEED]" << endl;
        MPI_Finalize();
        return 1;
    }

    string filename = args[0];
    int K = stoi(args[1]);
    string output_dir = args[2];
    int iters = (args.size() >= 4) ? stoi(args[3]) : 100;
    int chunks = (args.size() >= 5) ? stoi(args[4]) : -1;
    int seed = (args.size() >= 6) ? stoi(args[5]) : 42;

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

    int ret = starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, NULL);
    if (ret != 0) {
        if (rank == 0) cerr << "Error: Failed to initialize StarPU-MPI." << endl;
        MPI_Finalize();
        return 1;
    }

    KMeans kmeans(K, iters, output_dir, chunks, rank, size, dimensions, seed);
    kmeans.run(all_points, N);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    if (rank == 0) cout << "\nExecution time: " << duration.count() << " ms" << endl;

    print_starpu_worker_usage(rank);
    print_kernel_usage_metrics(rank);
    print_node_usage_metrics(rank, size);

    starpu_mpi_shutdown();

    if (rank == 0) {
        const char* sched = getenv("STARPU_SCHED");
        if (sched) cout << "Escalonador StarPU ativo: " << sched << endl;
    }

    MPI_Finalize();
    return 0;
}