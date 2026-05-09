/* ========================================================================== */
/* metrics_simple.cpp                                                          */
/* ========================================================================== */
/* Funções de métricas SEM dependência de MPI ou CUDA. Pode ser compilado     */
/* nas três versões (SEQ, OMP, StarPU). A versão StarPU também usa, em       */
/* conjunto com metrics.cpp (que tem extras MPI-específicos).                 */
/* ========================================================================== */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

using namespace std;

/* ========================================================================== */
/* Métricas padronizadas de execução (SSE + tempos + convergência)            */
/* ========================================================================== */
void print_execution_metrics(const std::string& versao_label,
                             const double* points, const int* labels,
                             const double* centroids,
                             int N, int K, int dims,
                             double t_total_ms, double t_useful_ms,
                             int iter_convergencia, int iter_max,
                             int mpi_ranks)
{
    // ----- Cálculo do SSE (Sum of Squared Errors) -----
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        int k = labels[i] - 1;   // labels são 1-based no projeto
        if (k >= 0 && k < K) {
            for (int d = 0; d < dims; d++) {
                double diff = points[i * dims + d] - centroids[k * dims + d];
                sse += diff * diff;
            }
        }
    }

    double t_ghost = t_total_ms - t_useful_ms;
    if (t_ghost < 0) t_ghost = 0.0;   // proteção contra jitter de relógio

    cout << "\n=== METRICAS DE EXECUCAO (" << versao_label << ") ===\n";
    cout << "[METRIC] versao="            << versao_label    << "\n";
    cout << "[METRIC] tempo_total_ms="    << t_total_ms      << "\n";
    cout << "[METRIC] tempo_util_ms="     << t_useful_ms     << "\n";
    cout << "[METRIC] tempo_ghost_ms="    << t_ghost         << "\n";
    cout << "[METRIC] iter_convergencia=" << iter_convergencia << "\n";
    cout << "[METRIC] iter_max="          << iter_max        << "\n";
    cout << "[METRIC] sse_final="         << fixed << setprecision(6) << sse << "\n";
    cout << "[METRIC] mpi_ranks="         << mpi_ranks       << "\n";
    cout << "==========================================\n";
    cout << defaultfloat;
}

/* ========================================================================== */
/* Wrapper SEQ                                                                */
/* ========================================================================== */
void compute_and_print_seq_metrics(
        const double* points, const int* labels, const double* centroids,
        int N, int K, int dims,
        int iter_convergencia, int iter_max,
        std::chrono::high_resolution_clock::time_point t_start,
        std::chrono::high_resolution_clock::time_point t_converge,
        std::chrono::high_resolution_clock::time_point t_end)
{
    using namespace std::chrono;
    double t_total  = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    double t_useful = duration_cast<microseconds>(t_converge - t_start).count() / 1000.0;

    print_execution_metrics("SEQ", points, labels, centroids,
                            N, K, dims,
                            t_total, t_useful,
                            iter_convergencia, iter_max,
                            /*mpi_ranks=*/1);
}

/* ========================================================================== */
/* Wrapper OMP                                                                */
/* ========================================================================== */
void compute_and_print_omp_metrics(
        const double* points, const int* labels, const double* centroids,
        int N, int K, int dims,
        int iter_convergencia, int iter_max, int mpi_ranks,
        std::chrono::high_resolution_clock::time_point t_start,
        std::chrono::high_resolution_clock::time_point t_converge,
        std::chrono::high_resolution_clock::time_point t_end)
{
    using namespace std::chrono;
    double t_total  = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    double t_useful = duration_cast<microseconds>(t_converge - t_start).count() / 1000.0;

    print_execution_metrics("OMP", points, labels, centroids,
                            N, K, dims,
                            t_total, t_useful,
                            iter_convergencia, iter_max, mpi_ranks);
}