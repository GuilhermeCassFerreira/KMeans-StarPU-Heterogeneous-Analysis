#include <starpu.h>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <iostream>

/* * IMPORTANTE: Esta estrutura deve ser IDÊNTICA à definida no main.cpp e no .cu.
 * Idealmente, mova isso para um header comum (ex: kmeans.h).
 */
struct HandleArgs {
    int K;
    int dimensions;
    int chunk_size;
};

// Contadores para métricas (referenciados via extern no main)
extern int cpu_assign_calls;
extern int cpu_calculate_calls;

/* -------------------------------------------------------------------------- */
/* TASKS DE NEGÓCIO (CPU)                                                     */
/* -------------------------------------------------------------------------- */

/*
 * CPU Task: assign_point_to_cluster
 * Calcula a distância de cada ponto para todos os centroides e define o rótulo.
 */
void assign_point_to_cluster_cpu(void *buffers[], void *cl_arg) {
    cpu_assign_calls++;

    HandleArgs *args = (HandleArgs *)cl_arg;
    
    // Buffers: [0] Pontos (R), [1] Centroides (R), [2] Labels (W)
    double *points = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *labels = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int K = args->K;
    int dim = args->dimensions;
    int n = args->chunk_size;

    for (int i = 0; i < n; i++) {
        double min_dist = DBL_MAX;
        int best_cluster = 0;

        // Para cada ponto, testar todos os clusters
        for (int c = 0; c < K; c++) {
            double dist = 0.0;
            // Distância Euclidiana ao quadrado
            for (int d = 0; d < dim; d++) {
                double diff = points[i * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        // Armazena ID base 1 (ou base 0 + 1) conforme sua lógica
        labels[i] = best_cluster + 1;
    }
}

/*
 * CPU Task: calculate_partial_sums
 * Lê os labels e acumula as coordenadas nos vetores de soma/contagem.
 */
void calculate_partial_sums_cpu(void *buffers[], void *cl_arg) {
    cpu_calculate_calls++;

    HandleArgs *args = (HandleArgs *)cl_arg;
    
    // Buffers: [0] Pontos, [1] Labels, [2] Somas(REDUX), [3] Counts(REDUX)
    double *points = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *labels = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    
    // No modo REDUX, o StarPU fornece um buffer local inicializado ou parcialmente preenchido.
    // Como esta task roda serialmente em um worker CPU, podemos acumular direto.
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int K = args->K;
    int dim = args->dimensions;
    int n = args->chunk_size;

    for (int i = 0; i < n; i++) {
        int cluster_id = labels[i] - 1; // Ajuste para índice 0

        // Verificação de sanidade
        if (cluster_id >= 0 && cluster_id < K) {
            partial_counts[cluster_id]++;
            
            for (int d = 0; d < dim; d++) {
                partial_sums[cluster_id * dim + d] += points[i * dim + d];
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/* FUNÇÕES DE REDUÇÃO (CPU)                                                   */
/* Usadas pelo StarPU para combinar buffers da CPU ou CPU+GPU                 */
/* -------------------------------------------------------------------------- */

// --- DOUBLE (Somas) ---

void redux_double_init_cpu(void *buffers[], void *cl_arg) {
    double *arr = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);
    
    // Inicializa com zero
    std::memset(arr, 0, n * sizeof(double));
}

void redux_double_reduce_cpu(void *buffers[], void *cl_arg) {
    double *dst = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *src = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);

    // Acumula src em dst
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

// --- INT (Contagens) ---

void redux_int_init_cpu(void *buffers[], void *cl_arg) {
    int *arr = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);
    
    // Inicializa com zero
    std::memset(arr, 0, n * sizeof(int));
}

void redux_int_reduce_cpu(void *buffers[], void *cl_arg) {
    int *dst = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *src = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int n = STARPU_VECTOR_GET_NX(buffers[0]);

    // Acumula src em dst
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}