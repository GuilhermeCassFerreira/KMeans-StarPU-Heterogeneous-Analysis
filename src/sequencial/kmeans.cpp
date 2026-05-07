#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include "../../include/kmeans_types.h"
#include "kmeans_seq.h"

bool read_points_from_file(const std::string &filename, std::vector<Point> &all_points, int &N, int &dimensions);

using namespace std;
using namespace chrono;

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        cout << "Uso: ./kmeans_seq <INPUT> <K> <OUT-DIR> [SEED]" << endl;
        return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    string output_dir = argv[3];
    int seed = (argc >= 5) ? stoi(argv[4]) : 42;

    vector<Point> all_points;
    int N = 0, dimensions = 0;
    if (!read_points_from_file(filename, all_points, N, dimensions)) {
        cout << "Erro: Falha ao abrir o arquivo." << endl;
        return 1;
    }

    if (N < K) {
        cout << "Erro: Número de clusters maior que o número de pontos." << endl;
        return 1;
    }

    // Alocação de arrays 1D para otimização de cache
    double *points = new double[N * dimensions];
    int *labels = new int[N];
    double *centroids = new double[K * dimensions];
    double *sums = new double[K * dimensions];
    int *counts = new int[K];

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dimensions; d++) {
            points[i * dimensions + d] = all_points[i].getVal(d);
        }
        labels[i] = 0;
    }

    // Inicialização determinística de centroides
    srand(seed);
    vector<int> chosen_indices;
    while ((int)chosen_indices.size() < K) {
        int r = rand() % N;
        if (find(chosen_indices.begin(), chosen_indices.end(), r) == chosen_indices.end()) {
            chosen_indices.push_back(r);
        }
    }

    for (int i = 0; i < K; ++i) {
        int point_idx = chosen_indices[i];
        for (int d = 0; d < dimensions; d++) {
            centroids[i * dimensions + d] = points[point_idx * dimensions + d];
        }
    }

    cout << "Clusters inicializados = " << K << "\n\n";
    cout << "Executando K-Means Clustering (Sequencial Otimizado)...\n";

    int iters = 100;
    auto start = high_resolution_clock::now();

    for (int iter = 1; iter <= iters; ++iter) {
        memset(sums, 0, K * dimensions * sizeof(double));
        memset(counts, 0, K * sizeof(int));

        // Chamada aos kernels
        int changes = assign_point_to_cluster_seq(points, centroids, labels, N, K, dimensions);
        calculate_partial_sums_seq(points, labels, sums, counts, N, K, dimensions);
        update_centroids_seq(sums, counts, centroids, K, dimensions);

        if (changes == 0 || iter >= iters) {
            cout << "Clustering concluído na iteração: " << iter << "\n\n";
            break;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Tempo de execução: " << duration.count() << " ms\n";

    // Salvamento de resultados
    string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());

    ofstream pointsFile(output_dir + "/" + to_string(K) + "-points.txt");
    for (int i = 0; i < N; i++) {
        pointsFile << labels[i] << "\n";
    }
    pointsFile.close();

    ofstream outfile(output_dir + "/" + to_string(K) + "-clusters.txt");
    outfile << fixed << setprecision(6);
    for (int k = 0; k < K; k++) {
        cout << "Cluster " << k+1 << " centroid: ";
        for (int d = 0; d < dimensions; d++) {
            cout << centroids[k * dimensions + d] << " ";
            outfile << centroids[k * dimensions + d] << " ";
        }
        cout << "\n";
        outfile << "\n";
    }
    outfile.close();

    // Limpeza
    delete[] points;
    delete[] labels;
    delete[] centroids;
    delete[] sums;
    delete[] counts;

    return 0;
}