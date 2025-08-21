#include <starpu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

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

// Estrutura para passar argumentos para o kernel StarPU
struct StarPUArgs {
    double *point_values;
    double *centroids;
    int K;
    int dimensions;
    int *nearestClusterId;
};

// Kernel para CPU
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
    if (argc != 4) {
        cout << "Error: command-line argument count mismatch. \n ./kmeans <INPUT> <K> <OUT-DIR>" << endl;
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

    starpu_init(NULL);

    auto start = high_resolution_clock::now();

    KMeans kmeans(K, iters, output_dir);
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