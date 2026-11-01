#ifndef KMEANS_RUNTIME_H
#define KMEANS_RUNTIME_H

#include <starpu.h>
#include <starpu_mpi.h>
#include <atomic>
#include <chrono>
#include "../../include/kmeans_types.h"
#include "../../include/options.h"
#include "../../include/metrics.h"

/* ========================================================================== */
/* Contadores globais de métricas                                             */
/* ========================================================================== */

extern int cpu_kernel_calls;
extern int cpu_assign_calls;
extern int cpu_calculate_calls;
extern int cpu_clean_calls;
extern int cpu_update_calls;
extern int cpu_accumulate_calls;

#ifdef STARPU_USE_CUDA
extern "C" {
    extern volatile int cuda_assign_calls;
    extern volatile int cuda_calculate_calls;
    extern volatile int cuda_clean_calls;
    extern volatile int cuda_update_calls;
    extern volatile int cuda_accumulate_calls;
    int get_cuda_kernel_calls();
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
bool read_points_flat(const std::string &filename, std::vector<double> &values, int &N, int &dimensions);

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
    std::vector<int> final_labels;   // rótulos finais (rank 0), para o SSE

    double *points_ptr;
    int *labels_ptr;
    std::vector<int> chunk_owners;
    std::vector<uint32_t> chunk_tamanhos;  // tamanho de cada chunk com chunks mistos (vazio = iguais)
    std::vector<char> chunk_grande;        // 1 = chunk grande, 0 = pequeno (chunks mistos)
    double t_loop_ms_ = 0.0;

    void clearClusters();
    int getChunkOwner(int chunk_id);

    int prioridadeChunk(int chunk_id);
    void submitTasks(int N, int it);

public:
    KMeans(int K, int iterations, std::string output_dir, int chunk_size, int rank, int size, int dims, int seed);

    // all_values: N*D doubles, preenchido só no rank 0
    void run(const std::vector<double> &all_values, int N);

    // Getters para o main calcular metricas apos a execucao
    const std::vector<double>& getCentroids() const { return centroids_data; }
    const std::vector<int>& getLabels() const { return final_labels; }
    int    getDimensions() const { return dimensions; }
    int    getK()          const { return K; }
    double getLoopMs()     const { return t_loop_ms_; }
};

#endif // KMEANS_RUNTIME_H