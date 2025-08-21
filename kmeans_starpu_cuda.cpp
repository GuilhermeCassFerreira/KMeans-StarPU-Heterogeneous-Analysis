#include <starpu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <atomic>
#include <iomanip>

using namespace std;
using namespace chrono;

// Variáveis globais para contagem de execuções
atomic<int> cpu_executions(0);
atomic<int> cuda_executions(0);

// Definir estrutura para argumentos das tarefas
struct task_args_struct {
    int num_points;
    int num_clusters;
    int dimensions;
    int start_idx;
    int end_idx;
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

// CUDA kernel para calcular distâncias e atribuir clusters
extern "C" void assign_points_cuda_kernel(void *buffers[], void *cl_arg);

// Kernel CPU para pontos menores com contagem
void assign_point_cpu(void *buffers[], void *cl_arg) {
    // Incrementar contador de execuções CPU
    cpu_executions++;
    
    float *points = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *centroids = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    int *assignments = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);
    
    // Fazer o casting usando a estrutura definida
    task_args_struct *args = (task_args_struct *)cl_arg;

    for (int p = args->start_idx; p < args->end_idx; p++) {
        float min_dist = INFINITY;
        int best_cluster = 0;

        for (int c = 0; c < args->num_clusters; c++) {
            float dist = 0.0f;
            for (int d = 0; d < args->dimensions; d++) {
                float diff = points[p * args->dimensions + d] - centroids[c * args->dimensions + d];
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        assignments[p] = best_cluster;
    }
}

// Wrapper para o kernel CUDA com contagem
void assign_point_cuda_wrapper(void *buffers[], void *cl_arg) {
    // Incrementar contador de execuções CUDA
    cuda_executions++;
    
    // Chamar o kernel CUDA real
    assign_points_cuda_kernel(buffers, cl_arg);
}

// Codelet híbrido (CPU + CUDA) com wrapper
static struct starpu_codelet kmeans_codelet = {
    .cpu_funcs = {assign_point_cpu},
    .cuda_funcs = {assign_point_cuda_wrapper},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W}
};

class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

    void printExecutionStats() {
        int total_executions = cpu_executions + cuda_executions;
        if (total_executions > 0) {
            double cpu_percentage = (cpu_executions.load() * 100.0) / total_executions;
            double cuda_percentage = (cuda_executions.load() * 100.0) / total_executions;
            
            cout << "\n================================" << endl;
            cout << "      EXECUTION STATISTICS" << endl;
            cout << "================================" << endl;
            cout << "Total tasks executed: " << total_executions << endl;
            cout << "CPU executions:   " << cpu_executions.load() << " (" << fixed << setprecision(2) << cpu_percentage << "%)" << endl;
            cout << "CUDA executions:  " << cuda_executions.load() << " (" << fixed << setprecision(2) << cuda_percentage << "%)" << endl;
            cout << "================================" << endl;
            
            // Análise da distribuição
            if (cuda_percentage > 50) {
                cout << "🚀 CUDA dominance: Excellent GPU utilization!" << endl;
            } else if (cuda_percentage > 25) {
                cout << "⚡ Balanced: Good hybrid CPU+GPU execution!" << endl;
            } else {
                cout << "💻 CPU dominance: Consider increasing task size for more GPU usage." << endl;
            }
            cout << "================================" << endl;
        }
    }

    void resetExecutionCounters() {
        cpu_executions = 0;
        cuda_executions = 0;
    }

    void assignPointsToClusters(vector<Point> &all_points) {
        // Preparar dados em formato contíguo
        vector<float> points_data(total_points * dimensions);
        vector<float> centroids_data(K * dimensions);
        vector<int> assignments(total_points);

        // Copiar dados dos pontos
        for (int i = 0; i < total_points; i++) {
            for (int d = 0; d < dimensions; d++) {
                points_data[i * dimensions + d] = (float)all_points[i].getVal(d);
            }
        }

        // Copiar dados dos centróides
        for (int i = 0; i < K; i++) {
            for (int d = 0; d < dimensions; d++) {
                centroids_data[i * dimensions + d] = (float)clusters[i].getCentroidByPos(d);
            }
        }

        // Registrar dados no StarPU
        starpu_data_handle_t points_handle, centroids_handle, assignments_handle;
        
        starpu_vector_data_register(&points_handle, STARPU_MAIN_RAM, 
                                   (uintptr_t)points_data.data(), 
                                   total_points * dimensions, sizeof(float));
        
        starpu_vector_data_register(&centroids_handle, STARPU_MAIN_RAM,
                                   (uintptr_t)centroids_data.data(),
                                   K * dimensions, sizeof(float));
        
        starpu_vector_data_register(&assignments_handle, STARPU_MAIN_RAM,
                                   (uintptr_t)assignments.data(),
                                   total_points, sizeof(int));

        // Estratégia otimizada para favorecer GPU em datasets maiores
        int base_tasks;
        if (total_points < 1000) {
            base_tasks = max(2, total_points / 500);  // Datasets pequenos: poucas tarefas
        } else if (total_points < 50000) {
            base_tasks = max(8, total_points / 5000); // Datasets médios: tarefas médias
        } else {
            base_tasks = max(16, total_points / 10000); // Datasets grandes: muitas tarefas
        }
        
        int num_tasks = min(static_cast<int>(starpu_worker_get_count() * 3), base_tasks);
        int points_per_task = total_points / num_tasks;

        cout << "Creating " << num_tasks << " tasks for " << total_points << " points" << endl;

        vector<struct starpu_task *> tasks(num_tasks);

        for (int t = 0; t < num_tasks; t++) {
            // Alocar estrutura de argumentos
            task_args_struct *task_args = new task_args_struct;

            task_args->num_points = total_points;
            task_args->num_clusters = K;
            task_args->dimensions = dimensions;
            task_args->start_idx = t * points_per_task;
            task_args->end_idx = (t == num_tasks - 1) ? total_points : (t + 1) * points_per_task;

            tasks[t] = starpu_task_create();
            tasks[t]->cl = &kmeans_codelet;
            tasks[t]->handles[0] = points_handle;
            tasks[t]->handles[1] = centroids_handle;
            tasks[t]->handles[2] = assignments_handle;
            tasks[t]->cl_arg = task_args;
            tasks[t]->cl_arg_size = sizeof(task_args_struct);

            // Opcional: prioridade para balancear CPU/GPU
            if (t % 2 == 0) {
                starpu_task_set_priority(tasks[t], STARPU_DEFAULT_PRIO + 1);
            }

            if (starpu_task_submit(tasks[t]) != 0) {
                cerr << "Error: Failed to submit StarPU task " << t << endl;
                return;
            }
        }

        // Esperar todas as tarefas
        starpu_task_wait_for_all();

        // Atualizar clusters dos pontos
        for (int i = 0; i < total_points; i++) {
            all_points[i].setCluster(assignments[i] + 1); // +1 para 1-based index
        }

        // Limpar handles
        starpu_data_unregister(points_handle);
        starpu_data_unregister(centroids_handle);
        starpu_data_unregister(assignments_handle);
    }

public:
    KMeans(int K, int iterations, string output_dir) {
        this->K = K;
        this->iters = iterations;
        this->output_dir = output_dir;
    }

    void run(vector<Point> &all_points) {
        resetExecutionCounters(); // Resetar contadores no início
        
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        cout << "Running K-Means with " << total_points << " points, " 
             << K << " clusters, " << dimensions << " dimensions" << endl;
        cout << "StarPU workers: " << starpu_worker_get_count() << endl;

        // Inicialização dos clusters
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

            // Atribuir pontos aos clusters usando StarPU (CPU + GPU)
            assignPointsToClusters(all_points);

            // Limpar clusters existentes
            clearClusters();

            // Reatribuir pontos aos clusters
            for (int i = 0; i < total_points; i++) {
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            // Recalcular centróides
            for (int i = 0; i < K; i++) {
                int clusterSize = clusters[i].getSize();
                for (int j = 0; j < dimensions; j++) {
                    double sum = 0.0;
                    if (clusterSize > 0) {
                        for (int p = 0; p < clusterSize; p++) {
                            sum += clusters[i].getPoint(p).getVal(j);
                        }
                        clusters[i].setCentroidByPos(j, sum / clusterSize);
                    }
                }
            }

            if (iter >= iters) {
                cout << "Clustering completed in iteration : " << iter << endl << endl;
                break;
            }
            iter++;
        }

        // Mostrar estatísticas de execução
        printExecutionStats();

        // Salvar resultados
        ofstream pointsFile;
        pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);
        if (pointsFile.is_open()) {
            for (int i = 0; i < total_points; i++) {
                pointsFile << all_points[i].getCluster() << endl;
            }
            pointsFile.close();
        }

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
            cout << "Error: Unable to write to clusters.txt" << endl;
        }
    }
};

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Error: command-line argument count mismatch. \n ./kmeans_cuda <INPUT> <K> <OUT-DIR>" << endl;
        return 1;
    }

    string output_dir = argv[3];
    int K = atoi(argv[2]);
    string filename = argv[1];
    ifstream infile(filename.c_str());

    if (!infile.is_open()) {
        cout << "Error: Failed to open file." << endl;
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

    if ((int)all_points.size() < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    int iters = 100;

    // Inicializar StarPU com suporte CUDA
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.ncuda = -1; // Usar todas as GPUs disponíveis (-1 = automático)
    conf.ncpus = -1; // Usar todos os CPUs disponíveis (-1 = automático)
    
    if (starpu_init(&conf) != 0) {
        cerr << "Error: Failed to initialize StarPU." << endl;
        return 1;
    }

    cout << "StarPU initialized successfully!" << endl;
    cout << "CUDA devices: " << starpu_cuda_worker_get_count() << endl;
    cout << "CPU workers: " << starpu_cpu_worker_get_count() << endl;

    // Início da medição de tempo
    auto start = high_resolution_clock::now();

    KMeans kmeans(K, iters, output_dir);
    kmeans.run(all_points);

    // Fim da medição de tempo
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Execution time: " << duration.count() << " ms" << endl;

    // Finalizar StarPU
    starpu_shutdown();

    return 0;
}