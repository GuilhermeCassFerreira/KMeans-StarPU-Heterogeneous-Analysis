#ifndef METRICS_SIMPLE_H
#define METRICS_SIMPLE_H

#include <string>
#include <chrono>

/* Função genérica — usada por SEQ, OMP e StarPU */
void print_execution_metrics(const std::string& versao_label,
                             const double* points, const int* labels,
                             const double* centroids,
                             int N, int K, int dims,
                             double t_total_ms, double t_useful_ms,
                             int iter_convergencia, int iter_max,
                             int mpi_ranks);

/* Wrapper SEQ */
void compute_and_print_seq_metrics(
        const double* points, const int* labels, const double* centroids,
        int N, int K, int dims,
        int iter_convergencia, int iter_max,
        std::chrono::high_resolution_clock::time_point t_start,
        std::chrono::high_resolution_clock::time_point t_converge,
        std::chrono::high_resolution_clock::time_point t_end);

/* Wrapper OMP */
void compute_and_print_omp_metrics(
        const double* points, const int* labels, const double* centroids,
        int N, int K, int dims,
        int iter_convergencia, int iter_max, int mpi_ranks,
        std::chrono::high_resolution_clock::time_point t_start,
        std::chrono::high_resolution_clock::time_point t_converge,
        std::chrono::high_resolution_clock::time_point t_end);

#endif // METRICS_SIMPLE_H