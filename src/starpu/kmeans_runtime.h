#ifndef KMEANS_RUNTIME_H
#define KMEANS_RUNTIME_H

#include <starpu.h>
#include <starpu_mpi.h>
#include <atomic>
#include <chrono>
#include "../../include/kmeans_types.h"
#include "../../include/options.h"
#include "../common/metrics_simple.h"

/* ========================================================================== */
/* Contadores globais de métricas                                             */
/* ========================================================================== */

extern int cpu_kernel_calls;
extern int cpu_assign_calls;
extern int cpu_calculate_calls;
extern int opencl_assign_calls;
extern int opencl_calculate_calls;

#ifdef __cplusplus
extern "C" {
#endif

extern int cuda_assign_calls;
extern int cuda_calculate_calls;

#ifdef STARPU_USE_CUDA
int get_cuda_kernel_calls();
void starpu_set_converged_cpu_ptr(int *ptr);
#endif

#ifdef __cplusplus
}
#endif

/* ========================================================================== */
/* Declarações das funções CPU (implementadas em kmeans_cpu.cpp)              */
/* ========================================================================== */

void assign_point_to_cluster_handles(void *buffers[], void *cl_arg);
void calculate_partial_sums(void *buffers[], void *cl_arg);
void clean_buffers_cpu(void *buffers[], void *cl_arg);
void update_centroids_cpu(void *buffers[], void *cl_arg);
void accumulate_nodes_cpu(void *buffers[], void *cl_arg);
void redux_double_init_cpu(void *buffers[], void *cl_arg);
void redux_double_reduce_cpu(void *buffers[], void *cl_arg);
void redux_int_init_cpu(void *buffers[], void *cl_arg);
void redux_int_reduce_cpu(void *buffers[], void *cl_arg);
void changes_var_init_cpu(void *buffers[], void *cl_arg);
void changes_var_reduce_cpu(void *buffers[], void *cl_arg);

/* ========================================================================== */
/* Declarações das funções CUDA (implementadas em kmeans_cuda.cu)            */
/* ========================================================================== */

#ifdef STARPU_USE_CUDA
#ifdef __cplusplus
extern "C" {
#endif

void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg);
void calculate_partial_sums_cuda(void *buffers[], void *cl_arg);
void clean_buffers_cuda(void *buffers[], void *cl_arg);
void update_centroids_cuda(void *buffers[], void *cl_arg);
void accumulate_nodes_cuda(void *buffers[], void *cl_arg);
void redux_double_init_cuda(void *buffers[], void *cl_arg);
void redux_double_reduce_cuda(void *buffers[], void *cl_arg);
void redux_int_init_cuda(void *buffers[], void *cl_arg);
void redux_int_reduce_cuda(void *buffers[], void *cl_arg);
void changes_var_init_cuda(void *buffers[], void *cl_arg);
void changes_var_reduce_cuda(void *buffers[], void *cl_arg);

#ifdef __cplusplus
}
#endif
#endif

/* ========================================================================== */
/* Codelets StarPU (definidas em kmeans_mpi.cpp)                             */
/* ========================================================================== */

extern struct starpu_codelet cl_assign_point_handles;
extern struct starpu_codelet cl_calculate_partial_sums;
extern struct starpu_codelet cl_clean_buffers;
extern struct starpu_codelet cl_update_centroids;
extern struct starpu_codelet cl_accumulate_nodes;

/* ========================================================================== */
/* Performance Models StarPU                                                  */
/* ========================================================================== */

extern struct starpu_perfmodel assign_perf_model;
extern struct starpu_perfmodel calculate_perf_model;
extern struct starpu_perfmodel clean_perf_model;
extern struct starpu_perfmodel update_perf_model;

/* ========================================================================== */
/* Declarações de io.cpp e metrics.cpp                                        */
/* ========================================================================== */

bool read_points_from_file(const std::string &filename, std::vector<Point> &all_points, int &N, int &dimensions);
void print_kernel_usage_metrics(int rank);
void print_starpu_worker_usage(int rank);
void print_node_usage_metrics(int rank, int world_size);

class KMeans;   // forward declaration (definição abaixo no mesmo header)

void compute_and_print_starpu_metrics(
        const KMeans& kmeans,
        const std::vector<Point>& all_points,
        int N, int iters, int mpi_ranks,
        std::chrono::high_resolution_clock::time_point t_start,
        std::chrono::high_resolution_clock::time_point t_end);

/* ========================================================================== */
/* Classe KMeans (implementada em kmeans_mpi.cpp)                            */
/* ========================================================================== */

class KMeans {
private:
    int K, iters, dimensions, total_points;
    std::vector<Cluster> clusters;
    std::string output_dir;
    int chunks;
    int seed;
    int mpi_rank, world_size;

    std::vector<double> points_data;
    std::vector<int> nearestClusterIds;
    starpu_data_handle_t points_handle;
    starpu_data_handle_t output_handle;
    std::vector<starpu_data_handle_t> points_children;
    std::vector<starpu_data_handle_t> outputs_children;
    int num_chunks;

    /* Buffers parciais por chunk — W exclusivo, calculate paralela */
    double *chunk_sums_ptr;
    int    *chunk_counts_ptr;
    std::vector<starpu_data_handle_t> chunk_sums_handle;
    std::vector<starpu_data_handle_t> chunk_counts_handle;

    /* Acumulador parcial por nó MPI */
    double *partial_sums_ptr;
    int *partial_counts_ptr;
    std::vector<starpu_data_handle_t> partial_sums_handle;
    std::vector<starpu_data_handle_t> partial_counts_handle;

    std::vector<double> centroids_data;
    starpu_data_handle_t centroids_handle;

    double *points_ptr;
    int *labels_ptr;
    std::vector<int> chunk_owners;

    void clearClusters();
    int getChunkOwner(int chunk_id);
    
    // Função unificada
    void submitTasks(int N, starpu_data_handle_t converged_handle, int *converged_flag_ptr, starpu_data_handle_t h_changes_handle);

public:
    KMeans(int K, int iterations, std::string output_dir, int chunk_size, int rank, int size, int dims, int seed);

    void run(std::vector<Point> &all_points, int N);

    // Getter para o main calcular SSE após a execução
    const std::vector<double>& getCentroids() const { return centroids_data; }
    int getDimensions() const { return dimensions; }
    int getK() const { return K; }
};

/* ========================================================================== */
/* Tracking de convergência (definidos em kmeans_mpi.cpp)                     */
/* ========================================================================== */
extern std::atomic<int>  g_iter_converged;        // -1 se não convergiu
extern std::atomic<bool> g_converge_captured;
extern std::chrono::high_resolution_clock::time_point g_t_converge;
extern std::chrono::high_resolution_clock::time_point g_t_start;

#endif // KMEANS_RUNTIME_H