#include <starpu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include "core_affinity.h"
#include <iomanip>
#include <cstdint>
#include <numeric>

#ifdef STARPU_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;
using namespace chrono;

#ifdef STARPU_USE_CUDA
extern "C" void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg);
extern "C" void calculate_partial_sums_cuda(void *buffers[], void *cl_arg);
extern "C" int get_cuda_kernel_calls();
#else
static int get_cuda_kernel_calls() { return 0; }
#endif

#ifdef STARPU_USE_OPENCL
extern "C" void assign_point_to_cluster_opencl(void *buffers[], void *cl_arg);
extern "C" int get_opencl_kernel_calls();
#else
static int get_opencl_kernel_calls() { return 0; }
#endif

static int cpu_kernel_calls = 0;
static int cpu_assign_calls = 0;
static int cpu_calculate_calls = 0;
extern "C" int cuda_assign_calls;
extern "C" int cuda_calculate_calls;
static int opencl_assign_calls = 0;
static int opencl_calculate_calls = 0;
bool use_heterogeneous_chunks = false;
bool init_on_gpu = false;
bool use_dynamic_chunks = false;
bool auto_adjust_chunks = false;

// Estrutura simplificada para argumentos (metadados)
struct HandleArgs {
    int K;
    int dimensions;
    int chunk_size;
};

// ...existing code...
void print_kernel_usage_metrics() {
    // Totais por kernel
    long total_assign = (long)cpu_assign_calls + (long)cuda_assign_calls + (long)opencl_assign_calls;
    long total_calculate = (long)cpu_calculate_calls + (long)cuda_calculate_calls + (long)opencl_calculate_calls;

    // Totais agregados
    long total_all = total_assign + total_calculate;

    auto pct = [](long part, long total)->double {
        if (total <= 0) return 0.0;
        return (double)part * 100.0 / (double)total;
    };

    cout << "Métricas de Execução:" << endl;
    cout << fixed << setprecision(1);

    // assign_point_to_cluster
    {
        long cpu = cpu_assign_calls;
        long cuda = cuda_assign_calls;
        long opencl = opencl_assign_calls;
        long tot = total_assign;
        cout << "- assign_point_to_cluster  (" << tot << " execs): ["
             << "CPU: " << pct(cpu, tot) << "% | "
             << "CUDA: " << pct(cuda, tot) << "% | "
             << "OpenCL: " << pct(opencl, tot) << "%]" << endl;
    }

    // calculate_partial_sums
    {
        long cpu = cpu_calculate_calls;
        long cuda = cuda_calculate_calls;
        long opencl = opencl_calculate_calls;
        long tot = total_calculate;
        cout << "- calculate_partial_sums   (" << tot << " execs): ["
             << "CPU: " << pct(cpu, tot) << "% | "
             << "CUDA: " << pct(cuda, tot) << "% | "
             << "OpenCL: " << pct(opencl, tot) << "%]" << endl;
    }

    // Totais agregados
    {
        long cpu = cpu_assign_calls + cpu_calculate_calls;
        long cuda = cuda_assign_calls + cuda_calculate_calls;
        long opencl = opencl_assign_calls + opencl_calculate_calls;
        long tot = total_all;
        cout << "- TOTAIS                   (" << tot << " execs): ["
             << "CPU: " << pct(cpu, tot) << "% | "
             << "CUDA: " << pct(cuda, tot) << "% | "
             << "OpenCL: " << pct(opencl, tot) << "%]" << endl;
    }

    cout << defaultfloat;
}

void print_starpu_worker_usage() {
    // Mantemos compatibilidade com chamada anterior — exibe somente resumo global (consistente)
    long cpu = cpu_kernel_calls;
    long cuda = get_cuda_kernel_calls();
    long opencl = get_opencl_kernel_calls();
    long total = cpu + cuda + opencl;

    cout << "Métricas de execução dos kernels:" << endl;
    cout << fixed << setprecision(1);

    if (total == 0) {
        cout << "CPU:    " << cpu << " vez(es) (0.0%)" << endl;
        cout << "GPU CUDA:   " << cuda << " vez(es) (0.0%)" << endl;
        cout << "OpenCL: " << opencl << " vez(es) (0.0%)" << endl;
        cout << "Total:  " << total << " chamadas" << endl;
        cout << defaultfloat;
        return;
    }

    auto pct = [&](long v)->double { return (double)v * 100.0 / (double)total; };

    cout << "CPU:    " << cpu << " vez(es) (" << pct(cpu) << "%)" << endl;
    cout << "GPU CUDA:   " << cuda << " vez(es) (" << pct(cuda) << "%)" << endl;
    cout << "OpenCL: " << opencl << " vez(es) (" << pct(opencl) << "%)" << endl;
    cout << "Total:  " << total << " chamadas" << endl;
    cout << defaultfloat;
}
// ...existing code...

// Kernel CPU usando handles
void assign_point_to_cluster_handles(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_assign_calls++;

    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) return;

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);
    if (!points_values || !centroids || !nearestClusterIds) return;

    int K = args->K;
    int dimensions = args->dimensions;
    int chunk_size = args->chunk_size;

    for (int idx = 0; idx < chunk_size; idx++) {
        double *point_values = points_values + idx * dimensions;
        double min_dist2 = numeric_limits<double>::max();
        int bestClusterId = -1;
        for (int i = 0; i < K; i++) {
            double dist2 = 0.0;
            for (int j = 0; j < dimensions; j++) {
                double diff = centroids[i * dimensions + j] - point_values[j];
                dist2 += diff * diff;
            }
            if (dist2 < min_dist2) {
                min_dist2 = dist2;
                bestClusterId = i;
            }
        }
        nearestClusterIds[idx] = bestClusterId + 1;
    }
}

// Kernel para calcular somas parciais e contagens
void calculate_partial_sums(void *buffers[], void *cl_arg) {
    cpu_calculate_calls++;


    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) {
        cerr << "[ERROR] Argumentos inválidos no kernel CPU calculate_partial_sums." << endl;
        return;
    }

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    if (!points_values || !nearestClusterIds || !partial_sums || !partial_counts) {
        cerr << "[ERROR] Buffers inválidos no kernel CPU calculate_partial_sums." << endl;
        return;
    }

    int K = args->K;
    int dimensions = args->dimensions;
    int chunk_size = args->chunk_size;

    // Inicializa somas e contagens parciais
    for (int c = 0; c < K; ++c) {
        for (int d = 0; d < dimensions; ++d) {
            partial_sums[c * dimensions + d] = 0.0;
        }
        partial_counts[c] = 0;
    }

    // Processa os pontos do chunk
    for (int idx = 0; idx < chunk_size; ++idx) {
        int cluster_id = nearestClusterIds[idx] - 1; // Ajusta para índice 0
        if (cluster_id >= 0 && cluster_id < K) {
            partial_counts[cluster_id]++;
            for (int d = 0; d < dimensions; ++d) {
                partial_sums[cluster_id * dimensions + d] += points_values[idx * dimensions + d];
            }
        }
    }
}

// Codelet com handles (3 buffers: pontos, centroids, saída)
static struct starpu_codelet cl_assign_point_handles = {
    .cpu_funcs = {assign_point_to_cluster_handles},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {assign_point_to_cluster_cuda},
#endif
#ifdef STARPU_USE_OPENCL
    .opencl_funcs = {assign_point_to_cluster_opencl},
#endif
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W}
};

// Codelet para cálculo parcial
static struct starpu_codelet cl_calculate_partial_sums = {
    .cpu_funcs = {calculate_partial_sums},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {calculate_partial_sums_cuda},
#endif
#ifdef STARPU_USE_OPENCL
    .opencl_funcs = {assign_point_to_cluster_opencl}, // se houver implementação OpenCL específica, ajustar
#endif
    .nbuffers = 4,
    .modes = {STARPU_R, STARPU_R, STARPU_W, STARPU_W}
};

class Point {
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

    vector<double> lineToVec(string &line) {
        vector<double> values;
        string tmp = "";

        for (int i = 0; i < (int)line.length(); i++) {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e') {
                tmp += line[i];
            } else if (tmp.length() > 0) {
                values.push_back(stod(tmp));
                tmp = "";
            }
        }
        if (tmp.length() > 0) {
            values.push_back(stod(tmp));
            tmp = "";
        }
        return values;
    }

public:
    Point(int id, string line) {
        pointId = id;
        values = lineToVec(line);
        dimensions = values.size();
        clusterId = 0;
    }

    int getDimensions() { return dimensions; }
    int getCluster() { return clusterId; }
    int getID() { return pointId; }
    void setCluster(int val) { clusterId = val; }
    double getVal(int pos) { return values[pos]; }
    void setValues(const vector<double>& vals) { values = vals; }
};

class Cluster {
private:
    int clusterId;
    vector<double> centroid;
    vector<Point> points;

public:
    Cluster(int clusterId, Point centroid) {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid.getDimensions(); i++) {
            this->centroid.push_back(centroid.getVal(i));
        }
        this->addPoint(centroid);
    }

    void addPoint(Point p) {
        p.setCluster(this->clusterId);
        points.push_back(p);
    }

    bool removePoint(int pointId) {
        int size = points.size();
        for (int i = 0; i < size; i++) {
            if (points[i].getID() == pointId) {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    void removeAllPoints() { points.clear(); }
    int getId() { return clusterId; }
    Point getPoint(int pos) { return points[pos]; }
    int getSize() { return points.size(); }
    double getCentroidByPos(int pos) { return centroid[pos]; }
    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }
};

class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    vector<double> device_times;
    string output_dir;
    int chunk_size;
    bool use_heterogeneous_chunks;

    vector<double> points_data;
    vector<int> nearestClusterIds;
    starpu_data_handle_t points_handle = nullptr;
    starpu_data_handle_t output_handle = nullptr;
    vector<starpu_data_handle_t> points_children;
    vector<starpu_data_handle_t> outputs_children;
    int num_chunks;

    vector<double> centroids_data;
    starpu_data_handle_t centroids_handle = nullptr;

#ifdef STARPU_USE_CUDA
    void *device_points_ptr = nullptr;
    void *device_output_ptr = nullptr;
#endif

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

    void assignPointsToClusters(vector<Point> &all_points) {
        int N = all_points.size();

        if ((int)centroids_data.size() != K * dimensions) centroids_data.resize(K * dimensions);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < dimensions; j++) {
                centroids_data[i * dimensions + j] = clusters[i].getCentroidByPos(j);
            }
        }

        if (!centroids_handle) {
            starpu_vector_data_register(&centroids_handle, STARPU_MAIN_RAM,
                                        (uintptr_t)centroids_data.data(), K * dimensions, sizeof(double));
            if (!centroids_handle) {
                cerr << "[ERROR] Registro de centroids_handle falhou (assignPointsToClusters fallback)." << endl;
                return;
            }
        }

        vector<HandleArgs*> allocated_args;
        allocated_args.reserve(num_chunks);

        if ((int)points_children.size() != num_chunks || (int)outputs_children.size() != num_chunks) {
            cerr << "[ERROR] points_children/outputs_children size mismatch num_chunks=" << num_chunks << endl;
            return;
        }

        vector<struct starpu_task*> pending;
        pending.reserve(num_chunks);
        int submitted = 0;
        auto overall_start = high_resolution_clock::now();

        for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
            int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
            if (this_chunk <= 0) break;

            HandleArgs *args = new HandleArgs{K, dimensions, this_chunk};
            allocated_args.push_back(args);

            struct starpu_task *task = starpu_task_create();
            task->cl = &cl_assign_point_handles;
            task->handles[0] = points_children[chunk_id];
            task->handles[1] = centroids_handle;
            task->handles[2] = outputs_children[chunk_id];
            task->cl_arg = args;
            task->cl_arg_size = sizeof(HandleArgs);

            int ret = starpu_task_submit(task);
            if (ret < 0) {
                cerr << "[ERROR] starpu_task_submit failed for chunk " << chunk_id << " ret=" << ret << endl;
                delete args;
                break;
            }
            pending.push_back(task);
            submitted++;
        }

        if (submitted > 0) {
            starpu_task_wait_for_all();
            auto overall_end = high_resolution_clock::now();
            double elapsed_total = duration_cast<milliseconds>(overall_end - overall_start).count() / 1000.0;
            double per_chunk = elapsed_total / (double)submitted;
            device_times.insert(device_times.end(), submitted, per_chunk);
        }

        for (auto a : allocated_args) delete a;
        allocated_args.clear();

        for (int i = 0; i < N; i++) {
            all_points[i].setCluster(nearestClusterIds[i]);
        }
    }

    void adjust_chunk_size_based_on_performance(vector<double> &device_times, int &chunk_size) {
        if (device_times.empty()) return;

        double avg_time = accumulate(device_times.begin(), device_times.end(), 0.0) / device_times.size();

        if (avg_time < 0.05) {
            chunk_size = min(chunk_size * 2, 8192);
        } else if (avg_time > 0.2) {
            chunk_size = max(chunk_size / 2, 128);
        }

        if (chunk_size <= 0) {
            cerr << "Error: chunk_size inválido. Ajustando para 1." << endl;
            chunk_size = 1;
        }

        num_chunks = (total_points + chunk_size - 1) / chunk_size;

        cout << "[INFO] Chunk size ajustado para " << chunk_size << ", num_chunks = " << num_chunks << endl;
    }

    void calculateCentroids(vector<Point> &all_points) {
        int N = all_points.size();

        vector<double> partial_sums(K * dimensions, 0.0);
        vector<int> partial_counts(K, 0);

        starpu_data_handle_t partial_sums_handle, partial_counts_handle;
        starpu_vector_data_register(&partial_sums_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)partial_sums.data(), K * dimensions, sizeof(double));
        starpu_vector_data_register(&partial_counts_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)partial_counts.data(), K, sizeof(int));

        vector<HandleArgs*> allocated_args;
        allocated_args.reserve(num_chunks);

        for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
            if (this_chunk <= 0) break;

            HandleArgs *args = new HandleArgs{K, dimensions, this_chunk};
            allocated_args.push_back(args);

            struct starpu_task *task = starpu_task_create();
            task->cl = &cl_calculate_partial_sums;
            task->handles[0] = points_children[chunk_id];
            task->handles[1] = outputs_children[chunk_id];
            task->handles[2] = partial_sums_handle;
            task->handles[3] = partial_counts_handle;
            task->cl_arg = args;
            task->cl_arg_size = sizeof(HandleArgs);

            int ret = starpu_task_submit(task);
            if (ret < 0) {
                cerr << "[ERROR] starpu_task_submit falhou para chunk " << chunk_id << " com retorno " << ret << endl;
                delete args;
                break;
            }
        }

        starpu_task_wait_for_all();

        vector<vector<double>> sums(K, vector<double>(dimensions, 0.0));
        vector<int> counts(K, 0);

        for (int c = 0; c < K; ++c) {
            counts[c] += partial_counts[c];
            for (int d = 0; d < dimensions; ++d) {
                sums[c][d] += partial_sums[c * dimensions + d];
            }
        }

        for (int c = 0; c < K; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < dimensions; ++d) {
                    clusters[c].setCentroidByPos(d, sums[c][d] / counts[c]);
                }
            }
        }

        starpu_data_unregister(partial_sums_handle);
        starpu_data_unregister(partial_counts_handle);

        for (auto a : allocated_args) delete a;
        allocated_args.clear();
    }

public:
    KMeans(int K, int iterations, string output_dir, int chunk_size, bool use_heterogeneous_chunks) {
        this->K = K;
        this->iters = iterations;
        this->output_dir = output_dir;
        this->chunk_size = chunk_size;
        this->use_heterogeneous_chunks = use_heterogeneous_chunks;
    }

    void run(vector<Point> &all_points) {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        int N = total_points;
        points_data.resize(N * dimensions);
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < dimensions; d++) {
                points_data[i * dimensions + d] = all_points[i].getVal(d);
            }
        }
        nearestClusterIds.assign(N, 0);

        points_handle = nullptr;
        output_handle = nullptr;
        starpu_vector_data_register(&points_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)points_data.data(), N * dimensions, sizeof(double));
        starpu_vector_data_register(&output_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)nearestClusterIds.data(), N, sizeof(int));
        if (points_handle == nullptr || output_handle == nullptr) {
            cerr << "[ERROR] Registro MAIN_RAM para points/output falhou." << endl;
            return;
        }

        unsigned workers = starpu_worker_get_count();
        int desired_tasks_per_worker = 4;
        int desired_num_chunks = max(1, (int)workers * desired_tasks_per_worker);
        chunk_size = max(1, (N + desired_num_chunks - 1) / desired_num_chunks);
        num_chunks = (N + chunk_size - 1) / chunk_size;

        struct starpu_data_filter points_filter = {
            .filter_func = starpu_vector_filter_block,
            .nchildren = (unsigned)num_chunks
        };
        struct starpu_data_filter output_filter = {
            .filter_func = starpu_vector_filter_block,
            .nchildren = (unsigned)num_chunks
        };
        starpu_data_partition(points_handle, &points_filter);
        starpu_data_partition(output_handle, &output_filter);

        points_children.resize(num_chunks);
        outputs_children.resize(num_chunks);
        for (int i = 0; i < num_chunks; ++i) {
            points_children[i] = starpu_data_get_child(points_handle, i);
            outputs_children[i] = starpu_data_get_child(output_handle, i);
        }

        clusters.clear();
        for (int i = 0; i < K; i++) {
            clusters.emplace_back(i + 1, all_points[i]);
        }

        centroids_data.resize(K * dimensions);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < dimensions; j++) {
                centroids_data[i * dimensions + j] = clusters[i].getCentroidByPos(j);
            }
        }
        centroids_handle = nullptr;
        starpu_vector_data_register(&centroids_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)centroids_data.data(), K * dimensions, sizeof(double));
        if (!centroids_handle) {
            cerr << "[WARN] Falha ao registrar centroids_handle persistente em run()." << endl;
        }

        for (int it = 0; it < iters; ++it) {
            clearClusters();
            assignPointsToClusters(all_points);

            if (use_dynamic_chunks) {
                adjust_chunk_size_based_on_performance(device_times, chunk_size);
                device_times.clear();

                starpu_data_unpartition(points_handle, STARPU_MAIN_RAM);
                starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);

                struct starpu_data_filter points_filter2 = {
                    .filter_func = starpu_vector_filter_block,
                    .nchildren = (unsigned)num_chunks
                };
                struct starpu_data_filter output_filter2 = {
                    .filter_func = starpu_vector_filter_block,
                    .nchildren = (unsigned)num_chunks
                };
                starpu_data_partition(points_handle, &points_filter2);
                starpu_data_partition(output_handle, &output_filter2);

                points_children.resize(num_chunks);
                outputs_children.resize(num_chunks);
                for (int i = 0; i < num_chunks; ++i) {
                    points_children[i] = starpu_data_get_child(points_handle, i);
                    outputs_children[i] = starpu_data_get_child(output_handle, i);
                }
            }

            calculateCentroids(all_points);

            for (int i = 0; i < K; i++) {
                for (int j = 0; j < dimensions; j++) {
                    centroids_data[i * dimensions + j] = clusters[i].getCentroidByPos(j);
                }
            }
        }

        starpu_task_wait_for_all();

        if (centroids_handle) {
            starpu_data_unregister(centroids_handle);
            centroids_handle = nullptr;
        }

        if (points_handle) starpu_data_unpartition(points_handle, STARPU_MAIN_RAM);
        if (output_handle) starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);
        if (points_handle) starpu_data_unregister(points_handle);
        if (output_handle) starpu_data_unregister(output_handle);
    }
};


int main(int argc, char **argv) {
    bool use_big_cores = false;

    vector<string> args;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--dynamic-chunks") use_dynamic_chunks = true;
        else if (arg == "--init-on-gpu") init_on_gpu = true;
        else if (arg == "--use-big-cores") use_big_cores = true;
        else if (arg == "--heterogeneous-chunks") use_heterogeneous_chunks = true;
        else if (arg == "--auto-adjust-chunks") auto_adjust_chunks = true;
        else args.push_back(arg);
    }

    if (args.size() < 3 || args.size() > 4) {
        cout << "Error: command-line argument count mismatch. \n ./kmeans <INPUT> <K> <OUT-DIR> [CHUNK_SIZE]" << endl;
        return 1;
    }

    string filename = args[0];
    int K = stoi(args[1]);
    string output_dir = args[2];
    int chunk_size = (args.size() == 4) ? stoi(args[3]) : -1;

    ifstream infile(filename.c_str());
    if (!infile.is_open()) {
        cout << "Error: Failed to open file: " << filename << endl;
        return 1;
    }

    int pointId = 1;
    vector<Point> all_points;
    string line;
    while (getline(infile, line)) {
        Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }
    infile.close();
    cout << "\nData fetched successfully!" << endl << endl;

    int N = all_points.size();
    if (N < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    int iters = 100;

    if (starpu_init(NULL) != 0) {
        cerr << "Error: Failed to initialize StarPU." << endl;
        return 1;
    }

    if (chunk_size == -1) {
        unsigned workers = starpu_worker_get_count();
        int desired_tasks_per_worker = 4;
        int desired_num_chunks = max(1, (int)workers * desired_tasks_per_worker);
        chunk_size = max(1, (int)((N + desired_num_chunks - 1) / desired_num_chunks));
    }

    cout << "chunk_size =" << chunk_size << endl;

    auto start = high_resolution_clock::now();
    KMeans kmeans(K, iters, output_dir, chunk_size, use_heterogeneous_chunks);
    kmeans.run(all_points);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Execution time: " << duration.count() << " ms" << endl;
    print_starpu_worker_usage();

    starpu_shutdown();

    const char* sched = getenv("STARPU_SCHED");
    if (sched)
        cout << "Escalonador StarPU ativo: " << sched << endl;
    else
        cout << "Escalonador StarPU padrão (ws) em uso." << endl;

    print_kernel_usage_metrics();

    return 0;
}