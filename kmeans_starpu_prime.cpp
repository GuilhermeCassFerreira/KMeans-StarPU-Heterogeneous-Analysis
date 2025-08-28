// ...existing code...
#include <starpu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <mpi.h>
#include <numeric>

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

// ... Point e Cluster classes inalteradas ...
// ...existing code...
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
    // Construtor padrão a partir de string (linha do arquivo)
    Point(int id, string line) {
        pointId = id;
        values = lineToVec(line);
        dimensions = values.size();
        clusterId = 0; // Inicialmente não atribuído a nenhum cluster
    }

    // Construtor alternativo a partir de vetor de valores
    Point(int id, const vector<double>& vals) {
        pointId = id;
        values = vals;
        dimensions = vals.size();
        clusterId = 0;
    }

    int getDimensions() { return dimensions; }
    int getCluster() { return clusterId; }
    int getID() { return pointId; }
    void setCluster(int val) { clusterId = val; }
    double getVal(int pos) { return values[pos]; }
};
// ...existing code...
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


// Estrutura para passar argumentos para o kernel StarPU
struct StarPUArgs {
    double *point_values;
    double *centroids;
    int K;
    int dimensions;
    int *nearestClusterId;
};

// ... assign_point_to_cluster e cl_assign_point inalterados ...
void assign_point_to_cluster(void *buffers[], void *cl_arg) {
    cpu_kernel_calls++;
    StarPUArgs *args = (StarPUArgs *)cl_arg;
    double *point_values = args->point_values;
    double *centroids = args->centroids;
    int K = args->K;
    int dimensions = args->dimensions;
    int *nearestClusterId = args->nearestClusterId;

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
    *nearestClusterId = bestClusterId + 1;
}

// Codelet StarPU com suporte CPU e CUDA
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

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

    void assignPointsToClusters(vector<Point> &all_points) {
        vector<int> nearestClusterIds(all_points.size());
        vector<struct starpu_task *> tasks(all_points.size());

        vector<double> centroids(K * dimensions);
        for (int i = 0; i < K; i++)
            for (int j = 0; j < dimensions; j++)
                centroids[i * dimensions + j] = clusters[i].getCentroidByPos(j);

        for (size_t i = 0; i < all_points.size(); i++) {
            double *point_values = new double[dimensions];
            for (int d = 0; d < dimensions; d++)
                point_values[d] = all_points[i].getVal(d);

            StarPUArgs *args = new StarPUArgs{
                point_values,
                centroids.data(),
                K,
                dimensions,
                &nearestClusterIds[i]
            };

            tasks[i] = starpu_task_create();
            tasks[i]->cl = &cl_assign_point;
            tasks[i]->cl_arg = args;
            tasks[i]->cl_arg_size = sizeof(StarPUArgs);
            starpu_task_submit(tasks[i]);
        }

        starpu_task_wait_for_all();

        for (size_t i = 0; i < all_points.size(); i++) {
            all_points[i].setCluster(nearestClusterIds[i]);
        }
    }

public:
    KMeans(int K, int iterations, string output_dir) {
        this->K = K;
        this->iters = iterations;
        this->output_dir = output_dir;
    }

    // Função modificada para MPI: clusters e centroides são sincronizados entre processos
    void run(vector<Point> &all_points, int mpi_rank, int mpi_size) {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        vector<int> used_pointIds;
        vector<double> centroids(K * dimensions);

        // Inicialização dos clusters apenas no rank 0
        if (mpi_rank == 0) {
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
            for (int i = 0; i < K; i++)
                for (int j = 0; j < dimensions; j++)
                    centroids[i * dimensions + j] = clusters[i].getCentroidByPos(j);
        }

        // Broadcast dos centroides iniciais para todos os processos
        MPI_Bcast(centroids.data(), K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Cada processo inicializa seus clusters locais com os centroides recebidos
        if (mpi_rank != 0) {
            clusters.clear();
            for (int i = 0; i < K; i++) {
                vector<double> vals(dimensions);
                for (int j = 0; j < dimensions; j++)
                    vals[j] = centroids[i * dimensions + j];
                Point p(-1, vals); // Use o construtor que recebe vetor de double
                Cluster cluster(i+1, p);
                for (int j = 0; j < dimensions; j++)
                    cluster.setCentroidByPos(j, vals[j]);
                clusters.push_back(cluster);
            }
        }

        if (mpi_rank == 0) {
            cout << "Clusters initialized = " << clusters.size() << endl << endl;
            cout << "Running K-Means Clustering.." << endl;
        }

        int iter = 1;
        while (true) {
            if (mpi_rank == 0)
                cout << "Iter - " << iter << "/" << iters << endl;

            // Broadcast centroides para todos os processos
            for (int i = 0; i < K; i++)
                for (int j = 0; j < dimensions; j++)
                    centroids[i * dimensions + j] = clusters[i].getCentroidByPos(j);
            MPI_Bcast(centroids.data(), K * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Atualiza centroides locais
            for (int i = 0; i < K; i++)
                for (int j = 0; j < dimensions; j++)
                    clusters[i].setCentroidByPos(j, centroids[i * dimensions + j]);

            assignPointsToClusters(all_points);
            clearClusters();

            for (int i = 0; i < (int)all_points.size(); i++) {
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            // Calcula somas locais para centroides
            vector<double> local_sums(K * dimensions, 0.0);
            vector<int> local_counts(K, 0);
            for (int i = 0; i < K; i++) {
                int ClusterSize = clusters[i].getSize();
                local_counts[i] = ClusterSize;
                for (int j = 0; j < dimensions; j++) {
                    double sum = 0.0;
                    if (ClusterSize > 0) {
                        for (int p = 0; p < ClusterSize; p++) {
                            sum += clusters[i].getPoint(p).getVal(j);
                        }
                    }
                    local_sums[i * dimensions + j] = sum;
                }
            }

            // Reduz somas e contagens para rank 0
            vector<double> global_sums(K * dimensions, 0.0);
            vector<int> global_counts(K, 0);
            MPI_Reduce(local_sums.data(), global_sums.data(), K * dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(local_counts.data(), global_counts.data(), K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            // Rank 0 atualiza centroides globais
            if (mpi_rank == 0) {
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < dimensions; j++) {
                        if (global_counts[i] > 0)
                            clusters[i].setCentroidByPos(j, global_sums[i * dimensions + j] / global_counts[i]);
                    }
                }
            }

            // Critério de parada simples: número fixo de iterações
            if (iter >= iters) {
                if (mpi_rank == 0)
                    cout << "Clustering completed in iteration : " << iter << endl << endl;
                break;
            }
            iter++;
        }

        // Coleta resultados dos clusters de todos os processos para o rank 0
        // Cada processo envia os clusters dos seus pontos
        vector<int> local_assignments(all_points.size());
        for (size_t i = 0; i < all_points.size(); i++)
            local_assignments[i] = all_points[i].getCluster();

        // Recolhe tamanhos
        vector<int> recvcounts(mpi_size), displs(mpi_size);
        int local_n = all_points.size();
        MPI_Gather(&local_n, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; i++)
                displs[i] = displs[i-1] + recvcounts[i-1];
        }

        vector<int> all_assignments;
        if (mpi_rank == 0)
            all_assignments.resize(accumulate(recvcounts.begin(), recvcounts.end(), 0));

        MPI_Gatherv(local_assignments.data(), local_n, MPI_INT,
                    all_assignments.data(), recvcounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        // Rank 0 salva arquivos de saída
        if (mpi_rank == 0) {
            ofstream pointsFile;
            pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);
            for (size_t i = 0; i < all_assignments.size(); i++) {
                pointsFile << all_assignments[i] << endl;
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
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc != 4) {
        if (mpi_rank == 0)
            cout << "Error: command-line argument count mismatch. \n ./kmeans <INPUT> <K> <OUT-DIR>" << endl;
        MPI_Finalize();
        return 1;
    }

    string output_dir = argv[3];
    int K = atoi(argv[2]);
    string filename = argv[1];

    vector<Point> all_points;
    int total_points = 0, dimensions = 0;

    // Rank 0 lê o arquivo e distribui os dados
    if (mpi_rank == 0) {
        ifstream infile(filename.c_str());
        if (!infile.is_open()) {
            cout << "Error: Failed to open file." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        string line;
        int pointId = 1;
        while (getline(infile, line)) {
            Point point(pointId, line);
            all_points.push_back(point);
            pointId++;
        }
        infile.close();
        total_points = all_points.size();
        if (total_points > 0)
            dimensions = all_points[0].getDimensions();
        cout << "\nData fetched successfully!" << endl << endl;
    }

    // Broadcast total_points e dimensions
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (total_points < K) {
        if (mpi_rank == 0)
            cout << "Error: Number of clusters greater than number of points." << endl;
        MPI_Finalize();
        return 1;
    }

    // Particiona os pontos entre os processos
    int points_per_proc = total_points / mpi_size;
    int remainder = total_points % mpi_size;
    int local_n = points_per_proc + (mpi_rank < remainder ? 1 : 0);
    vector<double> local_data(local_n * dimensions);

    // Rank 0 distribui os dados
    if (mpi_rank == 0) {
        vector<int> sendcounts(mpi_size), displs(mpi_size);
        int offset = 0;
        for (int i = 0; i < mpi_size; i++) {
            int n = points_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = n * dimensions;
            displs[i] = offset * dimensions;
            offset += n;
        }
        vector<double> all_data(total_points * dimensions);
        for (int i = 0; i < total_points; i++)
            for (int d = 0; d < dimensions; d++)
                all_data[i * dimensions + d] = all_points[i].getVal(d);
        MPI_Scatterv(all_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     local_data.data(), local_n * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     local_data.data(), local_n * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Cada processo monta seus próprios pontos
    if (mpi_rank != 0) {
        all_points.clear();
        for (int i = 0; i < local_n; i++) {
            string fake_line = "";
            for (int d = 0; d < dimensions; d++) {
                fake_line += to_string(local_data[i * dimensions + d]);
                if (d < dimensions - 1) fake_line += " ";
            }
            Point p(i+1, fake_line);
            all_points.push_back(p);
        }
    }

    int iters = 100;

    starpu_init(NULL);

    auto start = high_resolution_clock::now();

    KMeans kmeans(K, iters, output_dir);
    kmeans.run(all_points, mpi_rank, mpi_size);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    if (mpi_rank == 0) {
        cout << "Execution time: " << duration.count() << " ms" << endl;
        print_starpu_worker_usage();
    }

    starpu_shutdown();

    if (mpi_rank == 0) {
        const char* sched = getenv("STARPU_SCHED");
        if (sched)
            cout << "Escalonador StarPU ativo: " << sched << endl;
        else
            cout << "Escalonador StarPU padrão (ws) em uso." << endl;

        int n_cuda = get_cuda_kernel_calls();
        if (n_cuda > 0)
            cout << "O kernel CUDA foi executado " << n_cuda << " vez(es)!" << endl;
        else
            cout << "O kernel CUDA NÃO foi executado!" << endl;

        cout << "Métricas de execução dos kernels:" << endl;
        cout << "CPU:    " << cpu_kernel_calls << " vez(es)" << endl;
        cout << "CUDA:   " << get_cuda_kernel_calls() << " vez(es)" << endl;
        cout << "OpenCL: " << get_opencl_kernel_calls() << " vez(es)" << endl;
    }

    MPI_Finalize();
    return 0;
}
// ...existing code...