#include "kmeans_runtime.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iomanip> 
#include <fstream> 
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include "kmeans_mpi_tags.h"

using namespace std;
using namespace std::chrono;

/* ========================================================================== */
/* Callback do Update (rank 0)                                                */
/* ========================================================================== */
/* Sem detecção de convergência: as iterações rodam sempre até o máximo, e o  */
/* callback só imprime o progresso.                                           */

static std::atomic<int> g_update_calls{0};   // quantas tasks update já completaram
static int g_mpi_rank = 0;                    // rank MPI deste processo

extern "C" void update_centroids_callback(void *) {
    int call_idx = g_update_calls.fetch_add(1) + 1;
    if (g_mpi_rank == 0) fprintf(stderr, "[StarPU] Iteracao %d\n", call_idx);
}

/* ========================================================================== */
/* Performance Models (definições)                                            */
/* ========================================================================== */

struct starpu_perfmodel assign_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_assign_model_mpi"
};

struct starpu_perfmodel calculate_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_calculate_model_mpi"
};

struct starpu_perfmodel clean_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_clean_model_mpi"
};

struct starpu_perfmodel update_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_update_model_mpi"
};

/* Desligado com KMEANS_ACC_MODEL=0 (ver aplicar_opcoes_accumulate) */
struct starpu_perfmodel accumulate_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_accumulate_model_mpi"
};

/* ========================================================================== */
/* Codelets StarPU (definições atualizadas para Ghost Tasks)                  */
/* ========================================================================== */

struct starpu_codelet cl_assign_point_handles = {
    .cpu_funcs = {assign_point_to_cluster_handles},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {assign_point_to_cluster_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 3,
    /* points R, centroids R, labels RW */
    .modes = {STARPU_R, STARPU_R, STARPU_RW},
    .model = &assign_perf_model
};

struct starpu_codelet cl_calculate_partial_sums = {
    .cpu_funcs = {calculate_partial_sums},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {calculate_partial_sums_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 4,
    /* points R, labels R, chunk_sums W, chunk_counts W
     * W exclusivo por chunk → todas as 100 tasks rodam em paralelo sem contenção */
    .modes = {STARPU_R, STARPU_R, STARPU_W, STARPU_W},
    .model = &calculate_perf_model
};

struct starpu_codelet cl_clean_buffers = {
    .cpu_funcs = {clean_buffers_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {clean_buffers_cuda}, 
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 2,
    .modes = {STARPU_W, STARPU_W},
    .model = &clean_perf_model
};

struct starpu_codelet cl_update_centroids = {
    .cpu_funcs = {update_centroids_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {update_centroids_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 3,
    /* sums R, counts R, centroids RW */
    .modes = {STARPU_R, STARPU_R, STARPU_RW},
    .model = &update_perf_model
};

struct starpu_codelet cl_accumulate_nodes = {
    .cpu_funcs = {accumulate_nodes_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {accumulate_nodes_cuda},
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 4,
    /* Sem COMMUTE: ordem determinística das somas FP, mesmo resultado em toda execução */
    .modes = {STARPU_RW, STARPU_RW, STARPU_R, STARPU_R},
    .name = "kmeans_accumulate_mpi"
};

/* Soma das parciais dos outros nós no rank 0, só na CPU (desligada com KMEANS_ACC_NODES_CPU=0). As parciais
 * chegam pela rede na memória da CPU; somá-las na GPU exige copiar cada uma pelo PCIe, em série. */
struct starpu_codelet cl_accumulate_nodes_cpu = {
    .cpu_funcs = {accumulate_nodes_cpu},
    .nbuffers = 4,
    .modes = {STARPU_RW, STARPU_RW, STARPU_R, STARPU_R},
    .name = "kmeans_accumulate_nodes_cpu"
};

static bool g_acc_nodes_cpu = false;

/* Chunks mistos. KMEANS_CHUNKS_PEQUENOS=P (padrão 0 = desligado): além dos chunks do argumento,
 * que viram os "grandes", cada nó ganha P chunks pequenos com KMEANS_FRACAO_PEQUENOS (padrão 0,2)
 * dos seus pontos, depois dos grandes. Com chunks grandes iguais, uma tarefa de CPU dura boa parte
 * da iteração: quando um nó dá um chunk a mais para uma CPU, ele atrasa todos os outros (traces da
 * AWS, PQ com 32 nós: ~31 ms por iteração esperando o nó mais lento). Com chunks pequenos no fim,
 * esse erro custa um chunk pequeno. KMEANS_PRIORIDADE_GRANDES=1: Assign e Calculate dos chunks
 * grandes com prioridade máxima e dos pequenos com a mínima (só vale em escalonador com
 * prioridade, como o dmdas). */
static int    g_chunks_pequenos = 0;
static double g_fracao_pequenos = 0.2;
static bool   g_prioridade_grandes = false;
static bool   g_medir_nos = true;
/* KMEANS_RANK0_MENOS_PEQUENOS=k (com chunks mistos): o rank 0 fica com k chunks pequenos a menos,
 * e os nós 1..k com um a mais cada. O rank 0 também recebe as parciais, soma entre nós e faz o
 * Update; nos traces da AWS (PQ, 32 nós) ele calculava em média 7 ms mais devagar que os outros.
 * Os tamanhos dos chunks não mudam, então a calibração continua valendo. */
static int    g_rank0_menos_pequenos = 0;
/* KMEANS_COMMUTE (padrão 1): as somas no acumulador de cada nó em qualquer ordem. Em ordem fixa,
 * um chunk (ou nó) atrasado segura todas as somas seguintes, que depois rodam em fila (traces da
 * AWS, PQ com 2 nós: ~500 somas e ~25 ms por iteração depois do fim do cálculo). A ordem das
 * somas em ponto flutuante muda, então o SSE pode variar na última casa entre execuções. */
static bool   g_commute = true;

/* Valor 0/1 da variável de ambiente; sem a variável, vale o padrão */
static bool env_ligada(const char *nome, bool padrao) {
    const char *v = getenv(nome);
    return v ? atoi(v) == 1 : padrao;
}

/* Todas ligadas por padrão; =0 volta ao comportamento antigo, para medir cada uma.
 * KMEANS_ACC_MODEL: modelo de desempenho no Accumulate, para o dmda escolher entre CPU e GPU nas
 * somas locais de cada nó sabendo o tempo de cada uma (a parcial de um Calculate na GPU já está
 * na GPU). KMEANS_ACC_NODES_CPU: soma entre nós no rank 0 só na CPU. KMEANS_UPDATE_CPU: Update
 * só na CPU, com os centroides novos já na RAM para o envio aos outros nós. */
static void aplicar_opcoes_accumulate(int rank) {
    if (env_ligada("KMEANS_ACC_MODEL", true)) {
        cl_accumulate_nodes.model = &accumulate_perf_model;
        cl_accumulate_nodes_cpu.model = &accumulate_perf_model; // mesma função de CPU, mesmo modelo
    }
    g_acc_nodes_cpu = env_ligada("KMEANS_ACC_NODES_CPU", true);
    bool update_cpu = env_ligada("KMEANS_UPDATE_CPU", true);
    if (update_cpu) cl_update_centroids.where = STARPU_CPU;
    const char *peq = getenv("KMEANS_CHUNKS_PEQUENOS");
    const char *frac = getenv("KMEANS_FRACAO_PEQUENOS");
    g_chunks_pequenos = peq ? std::max(atoi(peq), 0) : 0;
    g_fracao_pequenos = frac ? atof(frac) : 0.2;
    g_prioridade_grandes = env_ligada("KMEANS_PRIORIDADE_GRANDES", false);
    g_medir_nos = env_ligada("KMEANS_MEDIR_NOS", true);
    const char *r0 = getenv("KMEANS_RANK0_MENOS_PEQUENOS");
    g_rank0_menos_pequenos = r0 ? std::max(atoi(r0), 0) : 0;
    g_commute = env_ligada("KMEANS_COMMUTE", true);
    starpu_data_access_mode acc_modo = g_commute ? (starpu_data_access_mode)(STARPU_RW | STARPU_COMMUTE) : STARPU_RW;
    cl_accumulate_nodes.modes[0] = cl_accumulate_nodes.modes[1] = acc_modo;
    cl_accumulate_nodes_cpu.modes[0] = cl_accumulate_nodes_cpu.modes[1] = acc_modo;
    if (rank == 0) {
        const char *blocos = getenv("KMEANS_ASSIGN_BLOCOS_SM");
        std::cout << "[ACCUMULATE] modelo de desempenho: " << (cl_accumulate_nodes.model ? "sim" : "nao")
                  << " | soma entre nos so na CPU: " << (g_acc_nodes_cpu ? "sim" : "nao")
                  << " | Update so na CPU: " << (update_cpu ? "sim" : "nao")
                  << " | Assign GPU balanceado: " << (env_ligada("KMEANS_ASSIGN_BALANCEADO", true) ? "sim" : "nao")
                  << " | blocos por SM: " << (blocos ? blocos : "2") << std::endl;
        std::cout << "[CHUNKS] pequenos por no: " << g_chunks_pequenos << " (fracao " << g_fracao_pequenos
                  << ") | rank 0 com menos pequenos: " << g_rank0_menos_pequenos
                  << " | somas em qualquer ordem: " << (g_commute ? "sim" : "nao")
                  << " | prioridade aos grandes: " << (g_prioridade_grandes ? "sim" : "nao")
                  << " | medicao por no: " << (g_medir_nos ? "sim" : "nao") << std::endl;
    }
}

/* ========================================================================== */
/* Medição por nó (KMEANS_MEDIR_NOS, padrão 1)                                */
/* ========================================================================== */
/* Por iteração, no relógio de cada nó: quando o primeiro Assign ficou pronto  */
/* (centroides chegaram), quando o último Accumulate local terminou (parcial   */
/* do nó pronta) e quantos Assign cada worker fez. No fim, o rank 0 junta os   */
/* nós e mostra quanto o mais lento atrasou os outros.                         */

struct MedidaTarefa { int iter; bool assign; };
static std::vector<MedidaTarefa> g_medidas;            // uma por (iteração, chunk, tipo)
static std::mutex g_medida_mtx;
static high_resolution_clock::time_point g_medida_t0;
static std::vector<double> g_t_pronto, g_t_parcial;   // por iteração, ms
static std::vector<int> g_assign_worker;              // [iteração * g_nworkers + worker]
static int g_nworkers = 0;

static double medida_agora() {
    return duration<double, milli>(high_resolution_clock::now() - g_medida_t0).count();
}

/* Prólogo do Assign: a tarefa ficou pronta (dependências satisfeitas) */
static void medida_pronta(void *arg) {
    auto *m = (MedidaTarefa *)arg;
    double t = medida_agora();
    std::lock_guard<std::mutex> l(g_medida_mtx);
    g_t_pronto[m->iter] = std::min(g_t_pronto[m->iter], t);
}

/* Fim do Assign (conta o worker) ou do Accumulate local (parcial do nó) */
static void medida_fim(void *arg) {
    auto *m = (MedidaTarefa *)arg;
    double t = medida_agora();
    int w = starpu_worker_get_id();
    std::lock_guard<std::mutex> l(g_medida_mtx);
    if (m->assign) {
        if (w >= 0 && w < g_nworkers) g_assign_worker[m->iter * g_nworkers + w]++;
    } else {
        g_t_parcial[m->iter] = std::max(g_t_parcial[m->iter], t);
    }
}

/* Junta as medidas no rank 0, imprime o resumo e grava <saída>/nos.csv */
static void reportar_nos(int rank, int size, int iters, const string &output_dir) {
    const int NV = 4; // duração, Assign na GPU, Assign na CPU, máximo de Assign num worker de CPU
    std::vector<double> local(iters * NV, 0.0);
    for (int it = 0; it < iters; it++) {
        int gpu = 0, cpu = 0, cpu_max = 0;
        for (int w = 0; w < g_nworkers; w++) {
            int c = g_assign_worker[it * g_nworkers + w];
            if (starpu_worker_get_type(w) == STARPU_CUDA_WORKER) gpu += c;
            else if (starpu_worker_get_type(w) == STARPU_CPU_WORKER) { cpu += c; cpu_max = std::max(cpu_max, c); }
        }
        bool ok = g_t_parcial[it] >= 0 && g_t_pronto[it] < 1e299;
        local[it * NV + 0] = ok ? g_t_parcial[it] - g_t_pronto[it] : -1.0;
        local[it * NV + 1] = gpu;
        local[it * NV + 2] = cpu;
        local[it * NV + 3] = cpu_max;
    }
    std::vector<double> todos(rank == 0 ? (size_t)size * iters * NV : 0);
    MPI_Gather(local.data(), iters * NV, MPI_DOUBLE, todos.data(), iters * NV, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) return;

    auto v = [&](int r, int it, int k) { return todos[((size_t)r * iters + it) * NV + k]; };
    string mk = "mkdir -p " + output_dir;
    if (system(mk.c_str()) != 0) cerr << "[AVISO] Falha ao criar " << output_dir << endl;
    ofstream csv(output_dir + "/nos.csv");
    csv << "iteracao,rank,calculo_ms,assign_gpu,assign_cpu,assign_cpu_max_worker\n";
    double soma_atraso = 0, max_atraso = 0, soma_med = 0, soma_lento = 0;
    double lento_gpu = 0, lento_cpumax = 0, med_gpu = 0, med_cpumax = 0;
    int n_it = 0;
    for (int it = 1; it < iters; it++) { // a 1ª iteração inclui a carga dos dados nas GPUs
        std::vector<std::pair<double, int>> d;
        for (int r = 0; r < size; r++) {
            csv << it << "," << r << "," << v(r, it, 0) << "," << v(r, it, 1) << "," << v(r, it, 2) << "," << v(r, it, 3) << "\n";
            if (v(r, it, 0) >= 0) d.push_back({v(r, it, 0), r});
        }
        if (d.empty()) continue;
        std::sort(d.begin(), d.end());
        auto med = d[(d.size() - 1) / 2];   // mediana de baixo: com 2 nós, o mais rápido
        auto lento = d.back();
        soma_atraso += lento.first - med.first;
        max_atraso = std::max(max_atraso, lento.first - med.first);
        soma_med += med.first;
        soma_lento += lento.first;
        lento_gpu += v(lento.second, it, 1); lento_cpumax += v(lento.second, it, 3);
        med_gpu += v(med.second, it, 1);     med_cpumax += v(med.second, it, 3);
        n_it++;
    }
    if (n_it == 0) return;
    cout << fixed << setprecision(2)
         << "[NOS] calculo por iteracao (centroides chegam -> parcial pronta): mediana dos nos " << soma_med / n_it
         << " ms, no mais lento " << soma_lento / n_it << " ms" << endl
         << "[NOS] atraso do no mais lento sobre a mediana: media " << soma_atraso / n_it
         << " ms, maximo " << max_atraso << " ms" << endl
         << "[NOS] Assign por iteracao, no mais lento: GPU " << lento_gpu / n_it << ", maximo num worker de CPU "
         << lento_cpumax / n_it << " | no mediano: GPU " << med_gpu / n_it << ", maximo num worker de CPU "
         << med_cpumax / n_it << defaultfloat << endl;
}

/* ========================================================================== */
/* Implementação da classe KMeans                                             */
/* ========================================================================== */

KMeans::KMeans(int K, int iterations, string output_dir, int chunks, int rank, int size, int dims, int seed)
    : K(K), iters(iterations), output_dir(output_dir), num_chunks(chunks), mpi_rank(rank),
      world_size(size), dimensions(dims), seed(seed),
      points_handle(nullptr), output_handle(nullptr),
      chunk_sums_ptr(nullptr), chunk_counts_ptr(nullptr),
      partial_sums_ptr(nullptr), partial_counts_ptr(nullptr),
      centroids_handle(nullptr), points_ptr(nullptr), labels_ptr(nullptr),
      total_points(0)
{
    g_mpi_rank = rank;
}

void KMeans::clearClusters() {
    for (int i = 0; i < K; i++) {
        clusters[i].removeAllPoints();
    }
}

int KMeans::getChunkOwner(int chunk_id) {
    if (!chunk_dono.empty()) return chunk_dono[chunk_id];  // chunks mistos: lista explícita
    // Blocos contíguos: rank r possui chunks [r*cpr, (r+1)*cpr)
    // Alinha com o scatter contíguo dos pontos em run()
    int cpr = num_chunks / std::max(world_size, 1);
    return std::min(chunk_id / std::max(cpr, 1), world_size - 1);
}

/* Prioridade das tarefas do chunk: máxima nos grandes e mínima nos pequenos com
 * KMEANS_PRIORIDADE_GRANDES=1; senão a padrão */
int KMeans::prioridadeChunk(int chunk_id) {
    if (!g_prioridade_grandes) return STARPU_DEFAULT_PRIO;
    return chunk_grande[chunk_id] ? starpu_sched_get_max_priority() : starpu_sched_get_min_priority();
}

void KMeans::submitTasks(int N, int it) {
    int zero = 0;
    void (*pronta)(void *) = g_medir_nos ? medida_pronta : NULL;
    void (*fim)(void *)    = g_medir_nos ? medida_fim : NULL;

    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int this_chunk = starpu_vector_get_nx(points_children[chunk_id]);
        if (this_chunk <= 0) break;
        MedidaTarefa *med = &g_medidas[((size_t)it * num_chunks + chunk_id) * 2];

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_assign_point_handles,
            STARPU_R,    points_children[chunk_id],
            STARPU_R,    centroids_handle,
            STARPU_RW,   outputs_children[chunk_id],
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &this_chunk, sizeof(int),
            STARPU_PRIORITY, prioridadeChunk(chunk_id),
            STARPU_PROLOGUE_CALLBACK, pronta,
            STARPU_PROLOGUE_CALLBACK_ARG_NFREE, med,
            STARPU_CALLBACK_WITH_ARG_NFREE, fim, med,
            0);
    }

    /* CLEAN: zera acumulador de cada nó antes das somas parciais */
    for (int n = 0; n < world_size; n++) {
        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_clean_buffers,
            STARPU_W, partial_sums_handle[n],
            STARPU_W, partial_counts_handle[n],
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &zero, sizeof(int),
            STARPU_EXECUTE_ON_NODE, n,
            0);
    }

    /* CALCULATE: cada chunk escreve em buffer PRÓPRIO (W, sem contenção).
     * Todas as 100 tasks rodam em paralelo. */
    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        int owners = chunk_owners[chunk_id];
        int this_chunk = starpu_vector_get_nx(points_children[chunk_id]);
        if (this_chunk <= 0) break;

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_calculate_partial_sums,
            STARPU_R, points_children[chunk_id],
            STARPU_R, outputs_children[chunk_id],
            STARPU_W, chunk_sums_handle[chunk_id],
            STARPU_W, chunk_counts_handle[chunk_id],
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_VALUE, &this_chunk, sizeof(int),
            STARPU_PRIORITY, prioridadeChunk(chunk_id),
            STARPU_EXECUTE_ON_NODE, owners,
            0);
    }

    /* ACCUMULATE local: chunk_sums[i] → partial_sums[owner] (COMMUTE) */
    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        int owners = chunk_owners[chunk_id];
        int this_chunk = starpu_vector_get_nx(points_children[chunk_id]);
        if (this_chunk <= 0) break;

        starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_accumulate_nodes,
            cl_accumulate_nodes.modes[0], partial_sums_handle[owners],
            cl_accumulate_nodes.modes[1], partial_counts_handle[owners],
            STARPU_R, chunk_sums_handle[chunk_id],
            STARPU_R, chunk_counts_handle[chunk_id],
            STARPU_VALUE, &K, sizeof(int),
            STARPU_VALUE, &dimensions, sizeof(int),
            STARPU_EXECUTE_ON_NODE, owners,
            STARPU_CALLBACK_WITH_ARG_NFREE, fim, &g_medidas[((size_t)it * num_chunks + chunk_id) * 2 + 1],
            0);
    }

    if (world_size > 1) {
            for (int n = 1; n < world_size; n++) {
                starpu_mpi_task_insert(MPI_COMM_WORLD, g_acc_nodes_cpu ? &cl_accumulate_nodes_cpu : &cl_accumulate_nodes,
                    cl_accumulate_nodes.modes[0], partial_sums_handle[0],
                    cl_accumulate_nodes.modes[1], partial_counts_handle[0],
                    STARPU_R, partial_sums_handle[n],    
                    STARPU_R, partial_counts_handle[n],  
                    STARPU_VALUE, &K, sizeof(int),
                    STARPU_VALUE, &dimensions, sizeof(int),
                    STARPU_EXECUTE_ON_NODE, 0,    
                    0);
            }
        }

        // A ATUALIZAÇÃO DOS CENTRÓIDES RODA SEMPRE, seja com 1 ou 100 nodos!
        // No rank 0, o callback imprime o progresso das iterações.
        if (mpi_rank == 0) {
            starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_update_centroids,
                STARPU_R,  partial_sums_handle[0],
                STARPU_R,  partial_counts_handle[0],
                STARPU_RW, centroids_handle,
                STARPU_VALUE, &K, sizeof(int),
                STARPU_VALUE, &dimensions, sizeof(int),
                STARPU_VALUE, &zero, sizeof(int),
                STARPU_EXECUTE_ON_NODE, 0,
                STARPU_CALLBACK, update_centroids_callback,
                0);
        } else {
            starpu_mpi_task_insert(MPI_COMM_WORLD, &cl_update_centroids,
                STARPU_R,  partial_sums_handle[0],
                STARPU_R,  partial_counts_handle[0],
                STARPU_RW, centroids_handle,
                STARPU_VALUE, &K, sizeof(int),
                STARPU_VALUE, &dimensions, sizeof(int),
                STARPU_VALUE, &zero, sizeof(int),
                STARPU_EXECUTE_ON_NODE, 0,
                0);
        }
    }

void KMeans::run(const vector<double> &all_values, int N) {
    aplicar_opcoes_accumulate(mpi_rank);
    total_points = N;

    // Chunks mistos: os chunks do argumento viram os grandes de cada nó, seguidos de
    // g_chunks_pequenos chunks pequenos com g_fracao_pequenos dos pontos do nó
    chunk_tamanhos.clear();
    chunk_dono.clear();
    chunk_grande.assign(num_chunks, 1);
    pontos_no.clear();
    if (g_chunks_pequenos > 0) {
        int grandes = std::max(num_chunks / world_size, 1);
        long local_n = N / world_size;
        long peq = (long)(g_fracao_pequenos * local_n) / g_chunks_pequenos;
        int menos = world_size > 1 ? g_rank0_menos_pequenos : 0;
        if (N % world_size != 0 || peq < 1 || peq * g_chunks_pequenos >= local_n ||
            menos > g_chunks_pequenos || menos > world_size - 1) {
            if (mpi_rank == 0) cerr << "[ERRO] Chunks mistos invalidos (N divisivel pelos nos, 0 < fracao < 1, "
                                       "rank 0 com menos pequenos <= pequenos e < nos)" << endl;
            exit(1);
        }
        long resto = local_n - peq * g_chunks_pequenos;
        chunk_grande.clear();
        pontos_no.assign(world_size, 0);
        for (int r = 0; r < world_size; r++) {
            for (int g = 0; g < grandes; g++) {
                uint32_t t = (uint32_t)(resto / grandes + (g < resto % grandes ? 1 : 0));
                chunk_tamanhos.push_back(t); chunk_grande.push_back(1); chunk_dono.push_back(r);
                pontos_no[r] += t;
            }
            int npeq = g_chunks_pequenos - (r == 0 ? menos : 0) + (r >= 1 && r <= menos ? 1 : 0);
            for (int p = 0; p < npeq; p++) {
                chunk_tamanhos.push_back((uint32_t)peq); chunk_grande.push_back(0); chunk_dono.push_back(r);
                pontos_no[r] += peq;
            }
        }
        num_chunks = (int)chunk_tamanhos.size();
        if (mpi_rank == 0)
            cout << "[CHUNKS] por no: " << grandes << " grandes de ~" << resto / grandes << " pontos + "
                 << g_chunks_pequenos << " pequenos de " << peq << " pontos (rank 0: "
                 << g_chunks_pequenos - menos << " pequenos; " << num_chunks << " chunks no total)" << endl;
    }
    points_ptr = nullptr;
    labels_ptr = nullptr;
    partial_sums_ptr = nullptr;
    partial_counts_ptr = nullptr;
    chunk_sums_ptr = nullptr;
    chunk_counts_ptr = nullptr;

    size_t points_bytes      = (size_t)N * dimensions * sizeof(double);
    size_t labels_bytes      = (size_t)N * sizeof(int);
    size_t sums_bytes        = (size_t)K * dimensions * sizeof(double);
    size_t counts_bytes      = (size_t)K * sizeof(int);
    size_t chunk_sums_bytes  = (size_t)num_chunks * K * dimensions * sizeof(double);
    size_t chunk_counts_bytes= (size_t)num_chunks * K * sizeof(int);

    if (starpu_malloc((void**)&points_ptr, points_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&labels_ptr, labels_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&partial_sums_ptr, sums_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&partial_counts_ptr, counts_bytes) != 0) exit(1);
    if (starpu_malloc((void**)&chunk_sums_ptr,  chunk_sums_bytes)  != 0) exit(1);
    if (starpu_malloc((void**)&chunk_counts_ptr, chunk_counts_bytes) != 0) exit(1);

    memset(labels_ptr, 0, labels_bytes);
    if (mpi_rank == 0) {
        memcpy(points_ptr, all_values.data(), points_bytes);
    } else {
        memset(points_ptr, 0, points_bytes);
    }

    // Distribui pontos: cada rank recebe sua fatia contígua (alinhada com getChunkOwner)
    // Rank r recebe points[r*local_n .. (r+1)*local_n - 1] e armazena no offset correto
    if (world_size > 1 && !pontos_no.empty()) {
        // Chunks mistos: cada nó recebe a soma dos seus chunks (o rank 0 pode ter menos)
        std::vector<int> counts(world_size), displs(world_size);
        long desl = 0;
        for (int r = 0; r < world_size; r++) {
            counts[r] = (int)(pontos_no[r] * dimensions);
            displs[r] = (int)desl;
            desl += (long)counts[r];
        }
        if (mpi_rank == 0) {
            MPI_Scatterv(points_ptr, counts.data(), displs.data(), MPI_DOUBLE,
                         MPI_IN_PLACE, counts[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                         points_ptr + displs[mpi_rank], counts[mpi_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    } else if (world_size > 1) {
        int local_n = N / world_size;
        int count   = local_n * dimensions;
        if (mpi_rank == 0) {
            MPI_Scatter(points_ptr, count, MPI_DOUBLE,
                        MPI_IN_PLACE, count, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(nullptr, count, MPI_DOUBLE,
                        points_ptr + (long)mpi_rank * count, count, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
        }
    }
    memset(partial_sums_ptr,  0, sums_bytes);
    memset(partial_counts_ptr,0, counts_bytes);
    memset(chunk_sums_ptr,    0, chunk_sums_bytes);
    memset(chunk_counts_ptr,  0, chunk_counts_bytes);

    starpu_vector_data_register(&points_handle, STARPU_MAIN_RAM, (uintptr_t)points_ptr, N, dimensions * sizeof(double));
    starpu_vector_data_register(&output_handle, STARPU_MAIN_RAM, (uintptr_t)labels_ptr, N, sizeof(int));

    starpu_mpi_data_register(points_handle, KMeansTags::POINTS, 0);
    starpu_mpi_data_register(output_handle, KMeansTags::LABELS, 0);

    partial_sums_handle.resize(world_size);
    partial_counts_handle.resize(world_size);

    for (int n = 0; n < world_size; n++) {
        if (n == mpi_rank) {
            starpu_vector_data_register(&partial_sums_handle[n], STARPU_MAIN_RAM, (uintptr_t)partial_sums_ptr, K * dimensions, sizeof(double));
            starpu_vector_data_register(&partial_counts_handle[n], STARPU_MAIN_RAM, (uintptr_t)partial_counts_ptr, K, sizeof(int));
        } else {
            starpu_vector_data_register(&partial_sums_handle[n], -1, (uintptr_t)NULL, K * dimensions, sizeof(double));
            starpu_vector_data_register(&partial_counts_handle[n], -1, (uintptr_t)NULL, K, sizeof(int));
        }
        starpu_mpi_data_register(partial_sums_handle[n], KMeansTags::PARTIAL_SUMS_BASE + n, n);
        starpu_mpi_data_register(partial_counts_handle[n], KMeansTags::PARTIAL_COUNTS_BASE + n, n);
    }

    struct starpu_data_filter filterChunks = {
            .filter_func = starpu_vector_filter_block,
            .nchildren = (unsigned)this->num_chunks 
        };

    if (!chunk_tamanhos.empty()) {
        filterChunks.filter_func = starpu_vector_filter_list;
        filterChunks.filter_arg_ptr = chunk_tamanhos.data();
    }
    starpu_data_partition(points_handle, &filterChunks);
    starpu_data_partition(output_handle, &filterChunks);

    chunk_owners.resize(num_chunks);
    points_children.resize(num_chunks);
    outputs_children.resize(num_chunks);
    chunk_sums_handle.resize(num_chunks);
    chunk_counts_handle.resize(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        chunk_owners[i] = getChunkOwner(i);

        points_children[i]  = starpu_data_get_child(points_handle, i);
        outputs_children[i] = starpu_data_get_child(output_handle, i);

        starpu_mpi_data_register(points_children[i],  KMeansTags::CHUNK_POINTS_BASE + i, chunk_owners[i]);
        starpu_mpi_data_register(outputs_children[i], KMeansTags::CHUNK_LABELS_BASE + i, chunk_owners[i]);

        starpu_vector_data_register(&chunk_sums_handle[i], STARPU_MAIN_RAM,
            (uintptr_t)(chunk_sums_ptr + (size_t)i * K * dimensions),
            K * dimensions, sizeof(double));
        starpu_vector_data_register(&chunk_counts_handle[i], STARPU_MAIN_RAM,
            (uintptr_t)(chunk_counts_ptr + (size_t)i * K),
            K, sizeof(int));
        starpu_mpi_data_register(chunk_sums_handle[i],   KMeansTags::CHUNK_SUMS_BASE   + i, chunk_owners[i]);
        starpu_mpi_data_register(chunk_counts_handle[i], KMeansTags::CHUNK_COUNTS_BASE + i, chunk_owners[i]);
    }

    centroids_data.resize(K * dimensions);

    if (mpi_rank == 0) {
        srand(this->seed);

        std::vector<int> chosen_indices;
        
        while ((int)chosen_indices.size() < K) {
            int r = rand() % N; 
            if (std::find(chosen_indices.begin(), chosen_indices.end(), r) == chosen_indices.end()) {
                chosen_indices.push_back(r);
            }
        }

        for (int i = 0; i < K; ++i) {
            int point_idx = chosen_indices[i];

            for (int j = 0; j < dimensions; j++) {
                centroids_data[i * dimensions + j] = all_values[(size_t)point_idx * dimensions + j];
            }
        }
        
        cout << "[INFO] Centroides iniciais escolhidos aleatoriamente (Seed: " << this->seed << ")" << endl;
    }

    MPI_Bcast(centroids_data.data(), K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *centroids_ptr_starpu = nullptr;
    size_t centroids_bytes = (size_t)K * dimensions * sizeof(double);
    if (starpu_malloc((void**)&centroids_ptr_starpu, centroids_bytes) != 0) exit(1);
    memcpy(centroids_ptr_starpu, centroids_data.data(), centroids_bytes);

    starpu_vector_data_register(&centroids_handle, STARPU_MAIN_RAM, (uintptr_t)centroids_ptr_starpu, K * dimensions, sizeof(double));
    starpu_mpi_data_register(centroids_handle, KMeansTags::CENTROIDS, 0);

    if (mpi_rank == 0) cout << "[INFO] Injetando max " << iters << " iteracoes no DAG (Assíncrono)..." << endl;

    auto t_loop_start = high_resolution_clock::now();

    g_nworkers = starpu_worker_get_count();
    g_t_pronto.assign(iters, 1e300);
    g_t_parcial.assign(iters, -1.0);
    g_assign_worker.assign((size_t)iters * g_nworkers, 0);
    g_medidas.resize((size_t)iters * num_chunks * 2);
    for (int it = 0; it < iters; ++it)
        for (int c = 0; c < num_chunks; ++c) {
            g_medidas[((size_t)it * num_chunks + c) * 2]     = {it, true};
            g_medidas[((size_t)it * num_chunks + c) * 2 + 1] = {it, false};
        }
    g_medida_t0 = t_loop_start;

    for (int it = 0; it < iters; ++it) {
        submitTasks(N, it);
    }

    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    t_loop_ms_ = duration<double, milli>(high_resolution_clock::now() - t_loop_start).count();
    if (g_medir_nos) reportar_nos(mpi_rank, world_size, iters, output_dir);

    // Antes do unpartition, em modo MPI distribuído precisamos trazer os
    // outputs (labels) dos chunks que pertencem a OUTROS ranks de volta para
    // o rank 0, que é quem vai consolidar tudo. Em world_size==1 isso é
    // desnecessário (e causa estado interno inconsistente que segfaulta o
    // unpartition), por isso protegemos com world_size>1.
    if (world_size > 1) {
        for (int i = 0; i < num_chunks; i++) {
            if (chunk_owners[i] != 0) {
                starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
                    outputs_children[i], 0, NULL, NULL);
            }
        }
        starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    }

    starpu_data_unpartition(points_handle, STARPU_MAIN_RAM);
    starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);

    if (mpi_rank == 0) {
        starpu_data_acquire(output_handle, STARPU_R);
        final_labels.assign(labels_ptr, labels_ptr + N);
        starpu_data_release(output_handle);
        
        cout << "[INFO] Salvando centróides finais..." << endl;

        starpu_data_acquire(centroids_handle, STARPU_R);

        string cmd = "mkdir -p " + output_dir;
        if (system(cmd.c_str()) != 0) {
            cerr << "[AVISO] Falha ao tentar criar o diretório: " << output_dir << endl;
        }

        string clusters_filename = output_dir + "/" + to_string(K) + "-clusters.txt";
        ofstream outfile(clusters_filename);

        if (!outfile.is_open()) {
            cerr << "[ERRO] Não foi possível criar o arquivo: " << clusters_filename << endl;
        } else {
            outfile << fixed << setprecision(6);
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < dimensions; j++) {
                    double val = centroids_ptr_starpu[i * dimensions + j];
                    outfile << val << " ";
                }
                outfile << endl;
            }
            outfile.close();
            cout << "[INFO] Arquivo de centróides salvo com sucesso em: " << clusters_filename << endl;
        }

        starpu_data_release(centroids_handle);
    }

    starpu_data_unregister(points_handle);
    starpu_data_unregister(output_handle);
    starpu_data_unregister(centroids_handle);
    memcpy(centroids_data.data(), centroids_ptr_starpu, centroids_bytes);
    for (int i = 0; i < num_chunks; i++) {
        starpu_data_unregister(chunk_sums_handle[i]);
        starpu_data_unregister(chunk_counts_handle[i]);
    }
    for (int n = 0; n < world_size; n++) {
        starpu_data_unregister(partial_sums_handle[n]);
        starpu_data_unregister(partial_counts_handle[n]);
    }

    starpu_free_noflag(points_ptr, points_bytes);
    starpu_free_noflag(labels_ptr, labels_bytes);
    starpu_free_noflag(chunk_sums_ptr,   chunk_sums_bytes);
    starpu_free_noflag(chunk_counts_ptr, chunk_counts_bytes);
    starpu_free_noflag(partial_sums_ptr, sums_bytes);
    starpu_free_noflag(partial_counts_ptr, counts_bytes);
    starpu_free_noflag(centroids_ptr_starpu, centroids_bytes);
}