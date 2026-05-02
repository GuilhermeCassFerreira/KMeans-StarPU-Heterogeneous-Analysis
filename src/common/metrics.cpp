#include <iostream>
#include <iomanip>
#include <vector>
#include <mpi.h>

// Variáveis CPU & OpenCL
extern int cpu_kernel_calls;
extern int cpu_assign_calls;
extern int cpu_calculate_calls;
extern int cpu_clean_calls;
extern int cpu_update_calls;
extern int cpu_accumulate_calls;

extern int opencl_assign_calls;
extern int opencl_calculate_calls;
extern int opencl_clean_calls;
extern int opencl_update_calls;
extern int opencl_accumulate_calls;

// Variáveis CUDA
#ifdef STARPU_USE_CUDA
extern "C" int get_cuda_kernel_calls();
extern "C" int cuda_assign_calls;
extern "C" int cuda_calculate_calls;
extern "C" int cuda_clean_calls;
extern "C" int cuda_update_calls;
extern "C" int cuda_accumulate_calls;
#else
static int get_cuda_kernel_calls() { return 0; }
static int cuda_assign_calls = 0;
static int cuda_calculate_calls = 0;
static int cuda_clean_calls = 0;
static int cuda_update_calls = 0;
static int cuda_accumulate_calls = 0;
#endif

using namespace std;

void print_kernel_usage_metrics(int rank) {
    if (rank != 0) return;

    long total_assign     = (long)cpu_assign_calls + (long)cuda_assign_calls + (long)opencl_assign_calls;
    long total_calculate  = (long)cpu_calculate_calls + (long)cuda_calculate_calls + (long)opencl_calculate_calls;
    long total_clean      = (long)cpu_clean_calls + (long)cuda_clean_calls + (long)opencl_clean_calls;
    long total_update     = (long)cpu_update_calls + (long)cuda_update_calls + (long)opencl_update_calls;
    long total_accumulate = (long)cpu_accumulate_calls + (long)cuda_accumulate_calls + (long)opencl_accumulate_calls;
    
    long total_all = total_assign + total_calculate + total_clean + total_update + total_accumulate;

    auto pct = [](long part, long total) -> double {
        if (total <= 0) return 0.0;
        return (double)part * 100.0 / (double)total;
    };

    cout << "\nMetricas de Execucao (No Mestre):" << endl;
    cout << fixed << setprecision(1);

    cout << "- assign_point_to_cluster  (" << total_assign << " execs): ["
         << "CPU: " << pct(cpu_assign_calls, total_assign) << "% | "
         << "CUDA: " << pct(cuda_assign_calls, total_assign) << "% | "
         << "OpenCL: " << pct(opencl_assign_calls, total_assign) << "%]" << endl;

    cout << "- calculate_partial_sums   (" << total_calculate << " execs): ["
         << "CPU: " << pct(cpu_calculate_calls, total_calculate) << "% | "
         << "CUDA: " << pct(cuda_calculate_calls, total_calculate) << "% | "
         << "OpenCL: " << pct(opencl_calculate_calls, total_calculate) << "%]" << endl;

    cout << "- clean_buffers            (" << total_clean << " execs): ["
         << "CPU: " << pct(cpu_clean_calls, total_clean) << "% | "
         << "CUDA: " << pct(cuda_clean_calls, total_clean) << "% | "
         << "OpenCL: " << pct(opencl_clean_calls, total_clean) << "%]" << endl;

    cout << "- update_centroids         (" << total_update << " execs): ["
         << "CPU: " << pct(cpu_update_calls, total_update) << "% | "
         << "CUDA: " << pct(cuda_update_calls, total_update) << "% | "
         << "OpenCL: " << pct(opencl_update_calls, total_update) << "%]" << endl;

    cout << "- accumulate_nodes         (" << total_accumulate << " execs): ["
         << "CPU: " << pct(cpu_accumulate_calls, total_accumulate) << "% | "
         << "CUDA: " << pct(cuda_accumulate_calls, total_accumulate) << "% | "
         << "OpenCL: " << pct(opencl_accumulate_calls, total_accumulate) << "%]" << endl;

    long sum_cpu = cpu_assign_calls + cpu_calculate_calls + cpu_clean_calls + cpu_update_calls + cpu_accumulate_calls;
    long sum_cuda = cuda_assign_calls + cuda_calculate_calls + cuda_clean_calls + cuda_update_calls + cuda_accumulate_calls;
    long sum_opencl = opencl_assign_calls + opencl_calculate_calls + opencl_clean_calls + opencl_update_calls + opencl_accumulate_calls;

    cout << "--------------------------------------------------------" << endl;
    cout << "- TOTAIS GERAIS            (" << total_all << " execs): ["
         << "CPU: " << pct(sum_cpu, total_all) << "% | "
         << "CUDA: " << pct(sum_cuda, total_all) << "% | "
         << "OpenCL: " << pct(sum_opencl, total_all) << "%]" << endl;

    cout << defaultfloat;
}

void print_starpu_worker_usage(int rank) {
    if (rank != 0) return;

    long cpu = cpu_kernel_calls;
    long cuda = get_cuda_kernel_calls();
    long total = cpu + cuda;

    cout << "\nMetricas de execucao dos kernels (Global da Maquina 0):" << endl;
    cout << fixed << setprecision(1);

    if (total == 0) {
        cout << "CPU:        " << cpu << " vez(es) (0.0%)" << endl;
        cout << "GPU CUDA:   " << cuda << " vez(es) (0.0%)" << endl;
        cout << "Total:      " << total << " chamadas" << endl;
        cout << defaultfloat;
        return;
    }

    auto pct = [&](long v) -> double { return (double)v * 100.0 / (double)total; };

    cout << "CPU:        " << cpu << " vez(es) (" << pct(cpu) << "%)" << endl;
    cout << "GPU CUDA:   " << cuda << " vez(es) (" << pct(cuda) << "%)" << endl;
    cout << "Total:      " << total << " chamadas" << endl;
    cout << defaultfloat;
}

void print_node_usage_metrics(int rank, int world_size) {
    long my_total_tasks = (long)cpu_assign_calls + (long)cpu_calculate_calls + 
                          (long)cpu_clean_calls + (long)cpu_update_calls + (long)cpu_accumulate_calls +
                          (long)cuda_assign_calls + (long)cuda_calculate_calls + 
                          (long)cuda_clean_calls + (long)cuda_update_calls + (long)cuda_accumulate_calls +
                          (long)opencl_assign_calls + (long)opencl_calculate_calls + 
                          (long)opencl_clean_calls + (long)opencl_update_calls + (long)opencl_accumulate_calls;

    vector<long> all_nodes_tasks;
    if (rank == 0) {
        all_nodes_tasks.resize(world_size);
    }

    MPI_Gather(&my_total_tasks, 1, MPI_LONG,
               all_nodes_tasks.data(), 1, MPI_LONG,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        long global_total_tasks = 0;
        for (int i = 0; i < world_size; i++) {
            global_total_tasks += all_nodes_tasks[i];
        }

        cout << "\nMetricas de Distribuicao entre Nodos MPI:" << endl;
        cout << fixed << setprecision(1);

        if (global_total_tasks == 0) {
            cout << "Nenhuma tarefa foi contabilizada." << endl;
        } else {
            for (int i = 0; i < world_size; i++) {
                double pct = (double)all_nodes_tasks[i] * 100.0 / (double)global_total_tasks;
                cout << "- Nodo " << i << ": " << all_nodes_tasks[i]
                     << " tarefas (" << pct << "%)" << endl;
            }
            cout << "- TOTAL  : " << global_total_tasks << " tarefas" << endl;
        }
        cout << defaultfloat;
    }
}