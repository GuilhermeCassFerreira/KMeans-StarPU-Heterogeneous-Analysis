#include <iostream>
#include <iomanip>
#include <vector>
#include <mpi.h>

#ifdef STARPU_USE_CUDA
extern "C" int get_cuda_kernel_calls();
#else
static int get_cuda_kernel_calls() { return 0; }
#endif

extern int cpu_kernel_calls;
extern int cpu_assign_calls;
extern int cpu_calculate_calls;
extern "C" int cuda_assign_calls;
extern "C" int cuda_calculate_calls;
extern int opencl_assign_calls;
extern int opencl_calculate_calls;

using namespace std;

void print_kernel_usage_metrics(int rank) {
    if (rank != 0) return;

    long total_assign = (long)cpu_assign_calls + (long)cuda_assign_calls + (long)opencl_assign_calls;
    long total_calculate = (long)cpu_calculate_calls + (long)cuda_calculate_calls + (long)opencl_calculate_calls;
    long total_all = total_assign + total_calculate;

    auto pct = [](long part, long total) -> double {
        if (total <= 0) return 0.0;
        return (double)part * 100.0 / (double)total;
    };

    cout << "Metricas de Execucao (No Mestre):" << endl;
    cout << fixed << setprecision(1);

    cout << "- assign_point_to_cluster  (" << total_assign << " execs): ["
         << "CPU: " << pct(cpu_assign_calls, total_assign) << "% | "
         << "CUDA: " << pct(cuda_assign_calls, total_assign) << "% | "
         << "OpenCL: " << pct(opencl_assign_calls, total_assign) << "%]" << endl;

    cout << "- calculate_partial_sums   (" << total_calculate << " execs): ["
         << "CPU: " << pct(cpu_calculate_calls, total_calculate) << "% | "
         << "CUDA: " << pct(cuda_calculate_calls, total_calculate) << "% | "
         << "OpenCL: " << pct(opencl_calculate_calls, total_calculate) << "%]" << endl;

    cout << "- TOTAIS                   (" << total_all << " execs): ["
         << "CPU: " << pct(cpu_assign_calls + cpu_calculate_calls, total_all) << "% | "
         << "CUDA: " << pct(cuda_assign_calls + cuda_calculate_calls, total_all) << "% | "
         << "OpenCL: " << pct(opencl_assign_calls + opencl_calculate_calls, total_all) << "%]" << endl;

    cout << defaultfloat;
}

void print_starpu_worker_usage(int rank) {
    if (rank != 0) return;

    long cpu = cpu_kernel_calls;
    long cuda = get_cuda_kernel_calls();
    long total = cpu + cuda;

    cout << "Metricas de execucao dos kernels (No Mestre):" << endl;
    cout << fixed << setprecision(1);

    if (total == 0) {
        cout << "CPU:    " << cpu << " vez(es) (0.0%)" << endl;
        cout << "GPU CUDA:   " << cuda << " vez(es) (0.0%)" << endl;
        cout << "Total:  " << total << " chamadas" << endl;
        cout << defaultfloat;
        return;
    }

    auto pct = [&](long v) -> double { return (double)v * 100.0 / (double)total; };

    cout << "CPU:    " << cpu << " vez(es) (" << pct(cpu) << "%)" << endl;
    cout << "GPU CUDA:   " << cuda << " vez(es) (" << pct(cuda) << "%)" << endl;
    cout << "Total:  " << total << " chamadas" << endl;
    cout << defaultfloat;
}

void print_node_usage_metrics(int rank, int world_size) {
    long my_total_tasks = (long)cpu_assign_calls + (long)cpu_calculate_calls +
                          (long)cuda_assign_calls + (long)cuda_calculate_calls +
                          (long)opencl_assign_calls + (long)opencl_calculate_calls;

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