#include <omp.h>
#include <stdio.h>

void assign_clusters_gpu(const double* points, double* centroids, int* labels, int N, int K, int D, int* changed) {
    static int offload_printed = 0;
    if (!offload_printed) {
        int on_device = 0;
        #pragma omp target map(tofrom: on_device)
        {
            on_device = !omp_is_initial_device();
            if (on_device) printf("[offload] running on device\n");
            else printf("[offload] running on host\n");
        }
        offload_printed = on_device ? 1 : -1;
    }


    int changes = 0;

    // Envia dados para GPU a cada chamada
    #pragma omp target teams distribute parallel for reduction(+:changes) \
            map(to: points[0:N*D], centroids[0:K*D]) \
            map(tofrom: labels[0:N])
    for (int i = 0; i < N; i++) {
        double min_dist_sq = 1e30;
        int best_cluster = 0;

        for (int k = 0; k < K; k++) {
            double dist_sq = 0.0;
            for (int d = 0; d < D; d++) {
                double diff = points[i * D + d] - centroids[k * D + d];
                dist_sq += diff * diff;
            }
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_cluster = k;
            }
        }

        if (labels[i] != best_cluster) {
            changes++;
            labels[i] = best_cluster;
        }
    }
    *changed = changes;
}

void update_centroids_gpu(const double* points, double* centroids, const int* labels, int N, int K, int D) {
    
    // Arrays temporários alocados diretamente no mapeamento
    // Precisamos de arrays para Soma e Contagem
    
    // NOTA: Como não estamos usando memória persistente, temos que mapear tudo
    // ou criar arrays temporários dentro do target region.
    // Usaremos mapeamento 'alloc' para criar buffers temporários na GPU.
    
    double* d_sums = new double[K * D];
    int* d_counts = new int[K];

    // Zera os buffers na GPU
    #pragma omp target teams distribute parallel for map(to: d_sums[0:K*D], d_counts[0:K])
    for(int i=0; i < K*D; i++) d_sums[i] = 0.0;
    
    #pragma omp target teams distribute parallel for map(to: d_counts[0:K])
    for(int i=0; i < K; i++) d_counts[i] = 0;

    // Passo 1: Acumulação na GPU
    #pragma omp target teams distribute parallel for \
            map(to: points[0:N*D], labels[0:N]) \
            map(tofrom: d_sums[0:K*D], d_counts[0:K])
    for (int i = 0; i < N; i++) {
        int c = labels[i];
        
        #pragma omp atomic
        d_counts[c]++;

        for (int d = 0; d < D; d++) {
            #pragma omp atomic
            d_sums[c * D + d] += points[i * D + d];
        }
    }

    // Passo 2: Cálculo da Média e atualização dos centróides
    // map(tofrom: centroids) garante que o resultado volta para a RAM
    #pragma omp target teams distribute parallel for \
            map(tofrom: centroids[0:K*D]) \
            map(to: d_sums[0:K*D], d_counts[0:K])
    for (int k = 0; k < K; k++) {
        if (d_counts[k] > 0) {
            for (int d = 0; d < D; d++) {
                centroids[k * D + d] = d_sums[k * D + d] / d_counts[k];
            }
        }
    }

    delete[] d_sums;
    delete[] d_counts;
}