#include <starpu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include "core_affinity.h" // Inclua o cabeçalho

using namespace std;
using namespace chrono;

#ifdef STARPU_USE_CUDA
extern "C" void assign_point_to_cluster_cuda(void *buffers[], void *cl_arg);
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
        clusterId = 0; // Initially not assigned to any cluster
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

// Estrutura para passar argumentos para o kernel StarPU (agora com chunking)
struct StarPUArgs {
    double *points_values;      // [chunk_size][dimensions] (flattened)
    int *nearestClusterIds;     // [chunk_size]
    double *centroids;          // [K][dimensions] (flattened)
    int K;
    int dimensions;
    int offset;                 // índice inicial no vetor global
    int chunk_size;             // número de pontos neste chunk
};

// Kernel para CPU (chunking)
void assign_point_to_cluster(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    StarPUArgs *args = (StarPUArgs *)cl_arg;
    double *points_values = args->points_values;
    double *centroids = args->centroids;
    int K = args->K;
    int dimensions = args->dimensions;
    int chunk_size = args->chunk_size;
    int *nearestClusterIds = args->nearestClusterIds;

    if (!points_values || !centroids || !nearestClusterIds || K <= 0 || dimensions <= 0 || chunk_size <= 0) {
        return;
    }

    for (int idx = 0; idx < chunk_size; idx++) {
        double *point_values = points_values + idx * dimensions;
        double min_dist = numeric_limits<double>::max();
        int bestClusterId = -1;

        for (int i = 0; i < K; i++) {
            double dist = 0.0;
            for (int j = 0; j < dimensions; j++) {
                double diff = centroids[i * dimensions + j] - point_values[j];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist < min_dist) {
                min_dist = dist;
                bestClusterId = i;
            }
        }
        nearestClusterIds[idx] = bestClusterId + 1;
    }
}

// Codelet StarPU com suporte CPU, CUDA e OpenCL
static struct starpu_codelet cl_assign_point = {
    .cpu_funcs = {assign_point_to_cluster},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {assign_point_to_cluster_cuda},
#endif
#ifdef STARPU_USE_OPENCL
    .opencl_funcs = {assign_point_to_cluster_opencl},
#endif
    .nbuffers = 0,
    .modes = {}
};

class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;
    int chunk_size;

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

    void assignPointsToClusters(vector<Point> &all_points) {
        int N = all_points.size();
        vector<int> nearestClusterIds(N);

        vector<double> centroids(K * dimensions);
        for (int i = 0; i < K; i++)
            for (int j = 0; j < dimensions; j++)
                centroids[i * dimensions + j] = clusters[i].getCentroidByPos(j);

        vector<struct starpu_task *> tasks;
        vector<StarPUArgs*> allocated_args; // guardamos ponteiros alocados para liberar após wait

        for (int offset = 0; offset < N; offset += chunk_size) {
            int this_chunk = std::min(chunk_size, N - offset);

            // Aloca bloco de pontos para o chunk (flattened)
            double *points_values = new double[this_chunk * dimensions];
            for (int i = 0; i < this_chunk; i++)
                for (int d = 0; d < dimensions; d++)
                    points_values[i * dimensions + d] = all_points[offset + i].getVal(d);

            // Aloca centroids para cada chunk (corrige possível ponteiro inválido)
            double *centroids_chunk = new double[K * dimensions];
            std::copy(centroids.begin(), centroids.end(), centroids_chunk);

            int *chunk_nearestClusterIds = nearestClusterIds.data() + offset;

            StarPUArgs *args = new StarPUArgs{
                points_values,
                chunk_nearestClusterIds,
                centroids_chunk,
                K,
                dimensions,
                offset,
                this_chunk
            };

            struct starpu_task *task = starpu_task_create();
            task->cl = &cl_assign_point;
            task->cl_arg = args;
            task->cl_arg_size = sizeof(StarPUArgs);
            tasks.push_back(task);
            allocated_args.push_back(args); // guardamos para liberar depois
            starpu_task_submit(task);
        }

        // Espera todas as tasks terminarem antes de usar/free
        starpu_task_wait_for_all();

        // Atualiza os clusters dos pontos
        for (int i = 0; i < N; i++) {
            all_points[i].setCluster(nearestClusterIds[i]);
        }

        // Libera memória alocada para args/chunks
        for (StarPUArgs* a : allocated_args) {
            delete[] a->points_values;
            delete[] a->centroids;
            delete a;
        }

        // Não chame starpu_task_destroy: StarPU destrói as tasks automaticamente (destroy=1, detach=1)
        tasks.clear();
        allocated_args.clear();
    }

public:
    KMeans(int K, int iterations, string output_dir, int chunk_size = 100) {
        this->K = K;
        this->iters = iterations;
        this->output_dir = output_dir;
        this->chunk_size = chunk_size;
    }

    void run(vector<Point> &all_points) {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        vector<int> used_pointIds;

        for (int i = 1; i <= K; i++) {
            while (true) {
                int index = rand() % total_points;
                if (find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end()) {
                    used_pointIds.push_back(index);
                    all_points[index].setCluster(i);
                    Cluster cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }
        cout << "Clusters initialized = " << clusters.size() << endl << endl;
        cout << "Running K-Means Clustering.." << endl;

        int iter = 1;
        while (true) {
            cout << "Iter - " << iter << "/" << iters << endl;
            bool done = true;

            assignPointsToClusters(all_points);
            clearClusters();

            for (int i = 0; i < total_points; i++) {
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            for (int i = 0; i < K; i++) {
                int ClusterSize = clusters[i].getSize();
                for (int j = 0; j < dimensions; j++) {
                    double sum = 0.0;
                    if (ClusterSize > 0) {
                        for (int p = 0; p < ClusterSize; p++) {
                            sum += clusters[i].getPoint(p).getVal(j);
                        }
                        clusters[i].setCentroidByPos(j, sum / ClusterSize);
                    }
                }
            }

            if (done || iter >= iters) {
                cout << "Clustering completed in iteration : " << iter << endl << endl;
                break;
            }
            iter++;
        }

        ofstream pointsFile;
        pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);

        for (int i = 0; i < total_points; i++) {
            pointsFile << all_points[i].getCluster() << endl;
        }

        pointsFile.close();

        ofstream outfile;
        outfile.open(output_dir + "/" + to_string(K) + "-clusters.txt");
        if (outfile.is_open()) {
            for (int i = 0; i < K; i++) {
                cout << "Cluster " << clusters[i].getId() << " centroid : ";
                for (int j = 0; j < dimensions; j++) {
                    cout << clusters[i].getCentroidByPos(j) << " ";
                    outfile << clusters[i].getCentroidByPos(j) << " ";
                }
                cout << endl;
                outfile << endl;
            }
            outfile.close();
        } else {
            cout << "Error: Unable to write to clusters.txt";
        }
    }
};

void print_starpu_worker_usage() {
    unsigned nworkers = starpu_worker_get_count();
    unsigned cpu_count = 0, cuda_count = 0, other_count = 0;
    cout << "Workers StarPU disponíveis:" << endl;
    for (unsigned i = 0; i < nworkers; i++) {
        enum starpu_worker_archtype type = starpu_worker_get_type(i);
        char name[64];
        starpu_worker_get_name(i, name, sizeof(name));        
        if (type == STARPU_CPU_WORKER) {
            cout << "Worker " << i << ": CPU (" << name << ")" << endl;
            cpu_count++;
        } else if (type == STARPU_CUDA_WORKER) {
            cout << "Worker " << i << ": GPU/CUDA (" << name << ")" << endl;
            cuda_count++;
        } else {
            cout << "Worker " << i << ": Outro tipo (" << name << ")" << endl;
            other_count++;
        }
    }
    cout << "Resumo: " << cpu_count << " CPU(s), " << cuda_count << " GPU(s), " << other_count << " outros." << endl;
    cout << "Se houver GPU(s) listada(s) acima, o StarPU está pronto para usá-las." << endl;
}

int main(int argc, char **argv) {
    // Variável para controlar o uso de big cores
    bool use_big_cores = false;

    // Parse de argumentos para flags
    vector<string> args;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--use-big-cores") {
            use_big_cores = true;
        } else {
            args.push_back(arg);
        }
    }

    // Verifique se os argumentos restantes são suficientes
    if (args.size() < 3 || args.size() > 4) {
        cout << "Error: command-line argument count mismatch. \n ./kmeans <INPUT> <K> <OUT-DIR> [CHUNK_SIZE]" << endl;
        return 1;
    }

    // Processa os argumentos principais
    string filename = args[0];
    int K = stoi(args[1]);
    string output_dir = args[2];
    int chunk_size = (args.size() == 4) ? stoi(args[3]) : 100; // padrão: 100

    // Debug: Verifique se a flag foi ativada
    if (use_big_cores) {
        cout << "Flag --use-big-cores detected." << endl;
        vector<int> big_cores = detect_big_cores();
        if (!set_affinity_with_fallback(big_cores)) {
            cerr << "Failed to set affinity. Proceeding with default settings." << endl;
        } else {
            cout << "Affinity successfully set." << endl;
        }
    }

    // Abrir o arquivo de entrada
    ifstream infile(filename.c_str());
    if (!infile.is_open()) {
        cout << "Error: Failed to open file: " << filename << endl;
        return 1;
    }

    // Lê os pontos do arquivo
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

    // Verifica se o número de clusters é válido
    if ((int)all_points.size() < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    // Configurações de iterações
    int iters = 100;

    // Inicializa o StarPU
    if (starpu_init(NULL) != 0) {
        cerr << "Error: Failed to initialize StarPU." << endl;
        return 1;
    }

    // Medição de tempo
    auto start = high_resolution_clock::now();

    // Executa o KMeans
    KMeans kmeans(K, iters, output_dir, chunk_size);
    kmeans.run(all_points);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Execution time: " << duration.count() << " ms" << endl;

    // Imprime informações sobre os workers do StarPU
    print_starpu_worker_usage();

    // Finaliza o StarPU
    starpu_shutdown();

    // Exibe o escalonador ativo
    const char* sched = getenv("STARPU_SCHED");
    if (sched)
        cout << "Escalonador StarPU ativo: " << sched << endl;
    else
        cout << "Escalonador StarPU padrão (ws) em uso." << endl;

    // Exibe métricas de execução dos kernels
    int n_cuda = get_cuda_kernel_calls();
    if (n_cuda > 0)
        cout << "O kernel CUDA foi executado " << n_cuda << " vez(es)!" << endl;
    else
        cout << "O kernel CUDA NÃO foi executado!" << endl;

    cout << "Métricas de execução dos kernels:" << endl;
    cout << "CPU:    " << cpu_kernel_calls << " vez(es)" << endl;
    cout << "CUDA:   " << get_cuda_kernel_calls() << " vez(es)" << endl;
    cout << "OpenCL: " << get_opencl_kernel_calls() << " vez(es)" << endl;

    return 0;
}