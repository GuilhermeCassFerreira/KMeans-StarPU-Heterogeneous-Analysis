#include <starpu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <cstdint>
#include <numeric>
#include <random>

#ifdef STARPU_USE_CUDA
#include <cuda_runtime.h>
#endif
#include <mpi.h>

using namespace std;
using namespace chrono;

#ifdef STARPU_USE_CUDA
extern "C" void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg);
extern "C" void calculate_partial_sums_cuda(void *buffers[], void *cl_arg);
extern "C" int get_cuda_kernel_calls();
#else
static int get_cuda_kernel_calls() { return 0; }
#endif

int cpu_kernel_calls = 0;
int cpu_assign_calls = 0;
int cpu_calculate_calls = 0;
extern "C" int cuda_assign_calls;
extern "C" int cuda_calculate_calls;
int opencl_assign_calls = 0;
int opencl_calculate_calls = 0;
bool use_heterogeneous_chunks = false;
bool init_on_gpu = false;
bool use_dynamic_chunks = false;
bool auto_adjust_chunks = false;

struct HandleArgs {
    int K;
    int dimensions;
    int chunk_size;
};

static struct starpu_perfmodel assign_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_assign_model" // O nome do arquivo salvo no disco
};

static struct starpu_perfmodel calculate_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_calculate_model"
};

static struct starpu_perfmodel clean_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_clean_model"
};

static struct starpu_perfmodel update_perf_model = {
    .type = STARPU_HISTORY_BASED,
    .symbol = "kmeans_update_model"
};

void print_kernel_usage_metrics() {
    long total_assign = (long)cpu_assign_calls + (long)cuda_assign_calls + (long)opencl_assign_calls;
    long total_calculate = (long)cpu_calculate_calls + (long)cuda_calculate_calls + (long)opencl_calculate_calls;
    long total_all = total_assign + total_calculate;

    auto pct = [](long part, long total)->double {
        if (total <= 0) return 0.0;
        return (double)part * 100.0 / (double)total;
    };

    cout << "Metricas de Execucao:" << endl;
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

void print_starpu_worker_usage() {
    long cpu = cpu_kernel_calls;
    long cuda = get_cuda_kernel_calls();
    long total = cpu + cuda;

    cout << "Metricas de execucao dos kernels:" << endl;
    cout << fixed << setprecision(1);

    if (total == 0) {
        cout << "CPU:    " << cpu << " vez(es) (0.0%)" << endl;
        cout << "GPU CUDA:   " << cuda << " vez(es) (0.0%)" << endl;
        cout << "Total:  " << total << " chamadas" << endl;
        cout << defaultfloat;
        return;
    }

    auto pct = [&](long v)->double { return (double)v * 100.0 / (double)total; };

    cout << "CPU:    " << cpu << " vez(es) (" << pct(cpu) << "%)" << endl;
    cout << "GPU CUDA:   " << cuda << " vez(es) (" << pct(cuda) << "%)" << endl;
    cout << "Total:  " << total << " chamadas" << endl;
    cout << defaultfloat;
}

void assign_point_to_cluster_handles(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    cpu_assign_calls++;

    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) return;

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);

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

void calculate_partial_sums(void *buffers[], void *cl_arg) {
    cpu_calculate_calls++;

    HandleArgs *args = (HandleArgs *)cl_arg;
    if (!args) return;

    double *points_values = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nearestClusterIds = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[3]);

    int K = args->K;
    int dimensions = args->dimensions;
    int chunk_size = args->chunk_size;

    for (int idx = 0; idx < chunk_size; ++idx) {
        int cluster_id = nearestClusterIds[idx] - 1;
        if (cluster_id >= 0 && cluster_id < K) {
            partial_counts[cluster_id]++;
            for (int d = 0; d < dimensions; ++d) {
                partial_sums[cluster_id * dimensions + d] += points_values[idx * dimensions + d];
            }
        }
    }
}

void clean_buffers_cpu(void *buffers[], void *cl_arg) {
    HandleArgs *args = (HandleArgs *)cl_arg;
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    
    memset(partial_sums, 0, args->K * args->dimensions * sizeof(double));
    memset(partial_counts, 0, args->K * sizeof(int));
}

void update_centroids_cpu(void *buffers[], void *cl_arg) {
    HandleArgs *args = (HandleArgs *)cl_arg;
    double *partial_sums = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *partial_counts = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
    double *centroids = (double *)STARPU_VECTOR_GET_PTR(buffers[2]);
    
    int K = args->K;
    int dim = args->dimensions;
    
    for (int c = 0; c < K; ++c) {
        if (partial_counts[c] > 0) {
            for (int d = 0; d < dim; ++d) {
                centroids[c * dim + d] = partial_sums[c * dim + d] / partial_counts[c];
            }
        }
    }
} 

// ==============================================================================
// ESTRUTURAS DOS CODELETS
// ==============================================================================
static struct starpu_codelet cl_assign_point_handles = {
    .cpu_funcs = {assign_point_to_cluster_handles},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {assign_point_to_cluster_cuda},
#endif
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .model = &assign_perf_model  // <--- LIGAÇÃO AQUI
};

static struct starpu_codelet cl_calculate_partial_sums = {
    .cpu_funcs = {calculate_partial_sums},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {calculate_partial_sums_cuda},
#endif
    .nbuffers = 4,
    .modes = {STARPU_R, STARPU_R, STARPU_RW, STARPU_RW},
    .model = &calculate_perf_model // <--- LIGAÇÃO AQUI
};

static struct starpu_codelet cl_clean_buffers = {
    .cpu_funcs = {clean_buffers_cpu},
    .nbuffers = 2,
    .modes = {STARPU_W, STARPU_W},
    .model = &clean_perf_model // <--- LIGAÇÃO AQUI
};

static struct starpu_codelet cl_update_centroids = {
    .cpu_funcs = {update_centroids_cpu},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .model = &update_perf_model // <--- LIGAÇÃO AQUI
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
    double *partial_sums_ptr = nullptr;
    int *partial_counts_ptr = nullptr;
    starpu_data_handle_t partial_sums_handle = nullptr;
    starpu_data_handle_t partial_counts_handle = nullptr;

    vector<double> centroids_data;
    starpu_data_handle_t centroids_handle = nullptr;

    double *points_ptr = nullptr;
    int *labels_ptr = nullptr;

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

    void assignPointsToClusters(vector<Point> &all_points) {
        int N = all_points.size();

        for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
            int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
            if (this_chunk <= 0) break;

            HandleArgs *args = (HandleArgs *)malloc(sizeof(HandleArgs));
            args->K = K;
            args->dimensions = dimensions;
            args->chunk_size = this_chunk;

            struct starpu_task *task = starpu_task_create();
            task->cl = &cl_assign_point_handles;
            task->handles[0] = points_children[chunk_id];
            task->handles[1] = centroids_handle;
            task->handles[2] = outputs_children[chunk_id];
            
            task->cl_arg = args;
            task->cl_arg_size = sizeof(HandleArgs);
            task->cl_arg_free = 1;

            starpu_task_submit(task);
        }
    }

    void calculateCentroids(vector<Point> &all_points) {
        int N = all_points.size();

        HandleArgs *clean_args = (HandleArgs *)malloc(sizeof(HandleArgs));
        clean_args->K = K;
        clean_args->dimensions = dimensions;
        clean_args->chunk_size = 0;

        struct starpu_task *task_clean = starpu_task_create();
        task_clean->cl = &cl_clean_buffers;
        task_clean->handles[0] = partial_sums_handle;
        task_clean->handles[1] = partial_counts_handle;
        task_clean->cl_arg = clean_args;
        task_clean->cl_arg_size = sizeof(HandleArgs);
        task_clean->cl_arg_free = 1;
        starpu_task_submit(task_clean);

        for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            int this_chunk = min(chunk_size, N - chunk_id * chunk_size);
            if (this_chunk <= 0) break;

            HandleArgs *task_args = (HandleArgs *)malloc(sizeof(HandleArgs));
            task_args->K = K;
            task_args->dimensions = dimensions;
            task_args->chunk_size = this_chunk;

            struct starpu_task *task = starpu_task_create();
            task->cl = &cl_calculate_partial_sums;
            task->handles[0] = points_children[chunk_id];
            task->handles[1] = outputs_children[chunk_id];
            task->handles[2] = partial_sums_handle; 
            task->handles[3] = partial_counts_handle;
            task->cl_arg = task_args;
            task->cl_arg_size = sizeof(HandleArgs);
            task->cl_arg_free = 1;
            starpu_task_submit(task);
        }

        HandleArgs *update_args = (HandleArgs *)malloc(sizeof(HandleArgs));
        update_args->K = K;
        update_args->dimensions = dimensions;
        update_args->chunk_size = 0;

        struct starpu_task *task_update = starpu_task_create();
        task_update->cl = &cl_update_centroids;
        task_update->handles[0] = partial_sums_handle;
        task_update->handles[1] = partial_counts_handle;
        task_update->handles[2] = centroids_handle;
        task_update->cl_arg = update_args;
        task_update->cl_arg_size = sizeof(HandleArgs);
        task_update->cl_arg_free = 1;
        starpu_task_submit(task_update);
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

        points_ptr = nullptr;
        labels_ptr = nullptr;
        partial_sums_ptr = nullptr;
        partial_counts_ptr = nullptr;

        size_t points_bytes = (size_t)N * dimensions * sizeof(double);
        size_t labels_bytes = (size_t)N * sizeof(int);
        size_t sums_bytes   = (size_t)K * dimensions * sizeof(double);
        size_t counts_bytes = (size_t)K * sizeof(int);

        if (starpu_malloc((void**)&points_ptr, points_bytes) != 0) { exit(1); }
        if (starpu_malloc((void**)&labels_ptr, labels_bytes) != 0) { exit(1); }
        if (starpu_malloc((void**)&partial_sums_ptr, sums_bytes) != 0) { exit(1); }
        if (starpu_malloc((void**)&partial_counts_ptr, counts_bytes) != 0) { exit(1); }

        for (int i = 0; i < N; i++) {
            labels_ptr[i] = 0;
            for (int d = 0; d < dimensions; d++) {
                points_ptr[i * dimensions + d] = all_points[i].getVal(d);
            }
        }

        points_handle = nullptr;
        output_handle = nullptr;
        partial_sums_handle = nullptr;
        partial_counts_handle = nullptr;

        starpu_vector_data_register(&points_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)points_ptr, N, dimensions * sizeof(double));
        starpu_vector_data_register(&output_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)labels_ptr, N, sizeof(int));
        starpu_vector_data_register(&partial_sums_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)partial_sums_ptr, K * dimensions, sizeof(double));
        starpu_vector_data_register(&partial_counts_handle, STARPU_MAIN_RAM,
                                    (uintptr_t)partial_counts_ptr, K, sizeof(int));

        unsigned workers = starpu_worker_get_count();
        int desired_tasks_per_worker = 4;
        int desired_num_chunks = max(1, (int)workers * desired_tasks_per_worker);
        
        if (this->chunk_size <= 0) {
            this->chunk_size = max(1, (N + desired_num_chunks - 1) / desired_num_chunks);
        }
        num_chunks = (N + this->chunk_size - 1) / this->chunk_size;
        this->chunk_size = (N + num_chunks - 1) / num_chunks;

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
        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int i = 0; i < K; ++i) {
            clusters.emplace_back(i + 1, all_points[indices[i]]);
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

        cout << "[INFO] A submeter o grafo de tarefas (DAG) para " << iters << " iteracoes..." << endl;
        auto overall_start = high_resolution_clock::now();

        for (int it = 0; it < iters; ++it) {
            assignPointsToClusters(all_points);
            calculateCentroids(all_points);
        }

        starpu_task_wait_for_all();
        
        auto overall_end = high_resolution_clock::now();
        double total_time_ms = duration_cast<milliseconds>(overall_end - overall_start).count();
        cout << "[INFO] Execucao do K-Means concluida em " << total_time_ms << " ms." << endl;

        if (points_handle) starpu_data_unpartition(points_handle, STARPU_MAIN_RAM);
        if (output_handle) starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);

        starpu_data_acquire(centroids_handle, STARPU_R);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < dimensions; j++) {
                clusters[i].setCentroidByPos(j, centroids_data[i * dimensions + j]);
            }
        }
        starpu_data_release(centroids_handle);

        starpu_data_acquire(output_handle, STARPU_R);
        for (int i = 0; i < N; i++) {
            all_points[i].setCluster(labels_ptr[i]);
        }
        starpu_data_release(output_handle);

        if (points_handle) starpu_data_unregister(points_handle);
        if (output_handle) starpu_data_unregister(output_handle);
        if (centroids_handle) starpu_data_unregister(centroids_handle);
        if (partial_sums_handle) starpu_data_unregister(partial_sums_handle);
        if (partial_counts_handle) starpu_data_unregister(partial_counts_handle);

        if (points_ptr) starpu_free_noflag(points_ptr, points_bytes);
        if (labels_ptr) starpu_free_noflag(labels_ptr, labels_bytes);
        if (partial_sums_ptr) starpu_free_noflag(partial_sums_ptr, sums_bytes);
        if (partial_counts_ptr) starpu_free_noflag(partial_counts_ptr, counts_bytes);
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
    if (sched) cout << "Escalonador StarPU ativo: " << sched << endl;
    else cout << "Escalonador StarPU padrao (ws) em uso." << endl;

    print_kernel_usage_metrics();
    return 0;
}