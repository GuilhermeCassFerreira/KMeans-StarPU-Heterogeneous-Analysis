#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>

using namespace std;

// --- CONFIGURAÇÃO ---
// 0 = GPU, 1 = CPU
#define RUN_MODE 0

// --- DECLARAÇÕES EXTERNAS ---
extern void assign_clusters_cpu(const double* points, double* centroids, int* labels, int N, int K, int D, int* changed);
extern void update_centroids_cpu(const double* points, double* centroids, const int* labels, int N, int K, int D);

extern void assign_clusters_gpu(const double* points, double* centroids, int* labels, int N, int K, int D, int* changed);
extern void update_centroids_gpu(const double* points, double* centroids, const int* labels, int N, int K, int D);

// Função auxiliar para leitura linearizada
vector<double> read_file_linear(string filename, int& N, int& D) {
    ifstream infile(filename);
    vector<double> data;
    if (!infile.is_open()) exit(1);
    string line, tmp;
    N = 0; D = 0;
    while (getline(infile, line)) {
        if(line.empty()) continue;
        int current_d = 0;
        tmp = "";
        for (char c : line) {
            if ((c >= '0' && c <= '9') || c == '.' || c == '+' || c == '-' || c == 'e') {
                tmp += c;
            } else if (!tmp.empty()) {
                data.push_back(stod(tmp));
                tmp = "";
                current_d++;
            }
        }
        if (!tmp.empty()) { data.push_back(stod(tmp)); current_d++; }
        if (N == 0) D = current_d;
        N++;
    }
    return data;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "./kmeans <INPUT> <K> <OUT-DIR>" << endl;
        return 1;
    }

    string filename = argv[1];
    int K = atoi(argv[2]);
    string output_dir = argv[3];
    int N, D;

    // 1. Leitura e Preparação
    cout << "Reading data..." << endl;
    vector<double> h_points = read_file_linear(filename, N, D);
    vector<int> h_labels(N, 0);
    vector<double> h_centroids(K * D);
    
    // Inicialização Aleatória
    srand(42);
    for (int k = 0; k < K; k++) {
        int idx = (rand() % N) * D;
        for (int d = 0; d < D; d++) h_centroids[k * D + d] = h_points[idx + d];
    }

    // Ponteiros crus
    double* ptr_points = h_points.data();
    double* ptr_centroids = h_centroids.data();
    int* ptr_labels = h_labels.data();

    cout << "Setup: " << N << " points, " << K << " clusters." << endl;
    cout << "Mode: " << (RUN_MODE ? "CPU" : "GPU") << endl;

    int max_iters = 100;
    int iter = 0;

    auto start = chrono::high_resolution_clock::now();

    // --- LOOP PRINCIPAL (SEQUENCIAL) ---
    // Nenhuma diretiva de dados aqui, conforme solicitado.
    while (iter < max_iters) {
        int changed = 0;

        #if RUN_MODE == 1
            // CPU
            assign_clusters_cpu(ptr_points, ptr_centroids, ptr_labels, N, K, D, &changed);
            update_centroids_cpu(ptr_points, ptr_centroids, ptr_labels, N, K, D);
        #else
            // GPU
            assign_clusters_gpu(ptr_points, ptr_centroids, ptr_labels, N, K, D, &changed);
            update_centroids_gpu(ptr_points, ptr_centroids, ptr_labels, N, K, D);
        #endif

        iter++;
        if (changed == 0) {
            cout << "Converged at iter " << iter << endl;
            break;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Time: " << elapsed.count() << "s" << endl;

    // 3. Escrita de Arquivos
    ofstream pfile(output_dir + "/" + to_string(K) + "-points.txt");
    for (int i = 0; i < N; i++) pfile << ptr_labels[i] + 1 << endl;
    pfile.close();

    ofstream cfile(output_dir + "/" + to_string(K) + "-clusters.txt");
    for (int k = 0; k < K; k++) {
        cfile << "Cluster " << k+1 << " centroid : ";
        for (int d = 0; d < D; d++) cfile << ptr_centroids[k*D + d] << " ";
        cfile << endl;
    }
    cfile.close();

    return 0;
}