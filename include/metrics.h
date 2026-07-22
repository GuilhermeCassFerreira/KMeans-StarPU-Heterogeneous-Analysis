#ifndef METRICS_H
#define METRICS_H

struct SeqMetrics {
    double t_loop_ms;
    double t_total_ms;
    int    iter_converged;
    int    iter_max;
    double sse;
};

struct OmpMetrics {
    double t_loop_ms;        // apenas o laco iterativo (exclusivo I/O e init)
    double t_total_ms;       // total com I/O final
    int    iter_converged;   // iteracoes executadas ate convergir (ou max)
    int    iter_max;
    double sse;
    int    mpi_ranks;
    int    omp_threads;
    int    mode;             // 0=CPU  1=GPU  2=Hibrido
    int    num_chunks;
    // contagens por operacao
    long   total_assign,    gpu_assign;
    long   total_calculate, gpu_calculate;
    long   total_update,    gpu_update;
};

struct StarPUMetrics {
    double t_loop_ms;
    double t_total_ms;
    int    iter_converged;
    int    iter_max;
    double sse;
    int    mpi_ranks;
    int    ncpu_workers;
    int    ncuda_workers;
    // contagens por kernel x device
    long   cpu_assign,      cuda_assign;
    long   cpu_calculate,   cuda_calculate;
    long   cpu_clean,       cuda_clean;
    long   cpu_update,      cuda_update;
    long   cpu_accumulate,  cuda_accumulate;
};

// declaradas em src/common/metrics.cpp
void print_omp_metrics   (int rank, int world_size, const OmpMetrics&    m);
void print_starpu_metrics(int rank, int world_size, const StarPUMetrics& m);

#endif // METRICS_H
