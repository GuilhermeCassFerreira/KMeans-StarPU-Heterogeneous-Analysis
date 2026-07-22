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

    // ---- Parsing de argumentos ----
    vector<string> args;
    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }

    if (args.size() < 3 || args.size() > 8) {
        if (rank == 0)
            cout << "Uso: ./kmeans_starpu <INPUT> <K> <OUT-DIR> [NUM_CHUNCK] [DYNAMIC_SCHED] [SEED] [INTERS]" << endl;
        MPI_Finalize();
        return 1;
    }

    string filename = args[0];
    int K = stoi(args[1]);
    string output_dir = args[2];
    int num_chunks    = (args.size() >= 4) ? stoi(args[3]) : -1;
    bool dynamic_sched = (args.size() >= 5) ? (stoi(args[4]) == 1) : false;
    int seed          = (args.size() >= 6) ? stoi(args[5]) : 42;
    int iters         = (args.size() >= 7) ? stoi(args[6]) : 100;
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (dynamic_sched) cout << "[MODO] Escalonamento DINAMICO (StarPU-MPI decide - Sem EXECUTE_ON_NODE)" << endl;
        else cout << "[MODO] Escalonamento ESTATICO (Manual via EXECUTE_ON_NODE)" << endl;
    }

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

    MPI_Bcast(&iters, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // ---- Inicialização do StarPU-MPI ----
    int ret = starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, NULL);
    if (ret != 0) {
        if (rank == 0) cerr << "Error: Failed to initialize StarPU-MPI." << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&num_chunks, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool use_heterogeneous_chunks_val = false;

    KMeans kmeans(K, iters, output_dir, num_chunks, rank, size, dimensions, seed);
    kmeans.run(all_points, N);

    auto end = high_resolution_clock::now();
    double t_total_ms = duration_cast<duration<double, milli>>(end - start).count();

    // ---- Metricas ----
    StarPUMetrics m{};
    m.t_loop_ms    = kmeans.getLoopMs();
    m.t_total_ms   = t_total_ms;
    m.iter_max     = iters;
    m.iter_converged = (g_iter_converged.load() > 0)
                       ? g_iter_converged.load() : iters;
    m.mpi_ranks    = size;

    m.ncpu_workers   = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    m.ncuda_workers  = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    m.cpu_assign     = cpu_assign_calls;
    m.cpu_calculate  = cpu_calculate_calls;
    m.cpu_clean      = cpu_clean_calls;
    m.cpu_update     = cpu_update_calls;
    m.cpu_accumulate = cpu_accumulate_calls;
#ifdef STARPU_USE_CUDA
    m.cuda_assign     = cuda_assign_calls;
    m.cuda_calculate  = cuda_calculate_calls;
    m.cuda_clean      = cuda_clean_calls;
    m.cuda_update     = cuda_update_calls;
    m.cuda_accumulate = cuda_accumulate_calls;
#endif

    // SSE (so no rank 0 que tem todos os pontos)
    if (rank == 0) {
        const auto& cents = kmeans.getCentroids();
        int dims = kmeans.getDimensions();
        int K_   = kmeans.getK();
        double sse = 0.0;
        for (int i = 0; i < N; i++) {
            int c = all_points[i].getCluster() - 1;
            if (c < 0 || c >= K_) continue;
            for (int d = 0; d < dims; d++) {
                double diff = all_points[i].getVal(d) - cents[c * dims + d];
                sse += diff * diff;
            }
        }
        m.sse = sse;
    }

    print_starpu_metrics(rank, size, m);

    // ---- Finalização ----
    starpu_mpi_shutdown();

    if (rank == 0) {
        const char* sched = getenv("STARPU_SCHED");
        if (sched) cout << "Escalonador StarPU ativo: " << sched << endl;
    }

    MPI_Finalize();
    return 0;
}