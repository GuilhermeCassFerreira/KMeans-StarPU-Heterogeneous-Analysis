#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace chrono;

// Função auxiliar para extrair as coordenadas da linha de texto
vector<double> lineToVec(const string &line) {
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
    }
    return values;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Error: ./kmeans_seq_otimizado <INPUT> <K> <OUT-DIR>" << endl;
        return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    string output_dir = argv[3];

    // ==========================================================
    // 1. LEITURA E ACHATAMENTO DE DADOS (Flattening para Array 1D)
    // ==========================================================
    ifstream infile(filename.c_str());
    if (!infile.is_open()) {
        cout << "Error: Failed to open file." << endl;
        return 1;
    }

    vector<vector<double>> raw_points;
    string line;
    while (getline(infile, line)) {
        raw_points.push_back(lineToVec(line));
    }
    infile.close();

    int N = raw_points.size();
    if (N < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }
    int dimensions = raw_points[0].size();

    // A MÁGICA DA LOCALIDADE DE CACHE: Alocação 1D contínua
    double *points = new double[N * dimensions];
    int *labels = new int[N];
    double *centroids = new double[K * dimensions];
    double *sums = new double[K * dimensions];
    int *counts = new int[K];

    // Transferindo do vector temporário para o array 1D
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dimensions; d++) {
            points[i * dimensions + d] = raw_points[i][d];
        }
        labels[i] = 0;
    }
    raw_points.clear(); // Limpa a memória temporária

    // ==========================================================
    // 2. INICIALIZAÇÃO DETERMINÍSTICA (Semente = 42)
    // ==========================================================
    srand(42); 
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

    cout << "Clusters initialized = " << K << "\n\n";
    cout << "Running K-Means Clustering (Optimized Sequential)..\n";

    int iters = 100;

    // ==========================================================
    // 3. LAÇO PRINCIPAL OTIMIZADO (Somas Parciais e Sem sqrt/pow)
    // ==========================================================
    auto start = high_resolution_clock::now();

    for (int iter = 1; iter <= iters; ++iter) {
        // Zera os acumuladores a cada iteração
        memset(sums, 0, K * dimensions * sizeof(double));
        memset(counts, 0, K * sizeof(int));
        
        bool done = true;

        // FASE DE ATRIBUIÇÃO E ACÚMULO (FUSÃO DE LAÇOS)
        for (int i = 0; i < N; i++) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = -1;

            // Acha o centróide mais próximo
            for (int k = 0; k < K; k++) {
                double dist = 0.0;
                for (int d = 0; d < dimensions; d++) {
                    double diff = points[i * dimensions + d] - centroids[k * dimensions + d];
                    dist += diff * diff; // Muito mais rápido que pow()
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            // Verifica convergência
            if (labels[i] != (best_cluster + 1)) {
                labels[i] = best_cluster + 1;
                done = false;
            }

            // SOMAS PARCIAIS (Acumula na mesma hora, evitando ler a RAM duas vezes)
            counts[best_cluster]++;
            for (int d = 0; d < dimensions; d++) {
                sums[best_cluster * dimensions + d] += points[i * dimensions + d];
            }
        }

        // FASE DE ATUALIZAÇÃO (Divide as somas para achar o novo meio)
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                for (int d = 0; d < dimensions; d++) {
                    centroids[k * dimensions + d] = sums[k * dimensions + d] / counts[k];
                }
            }
        }

        if (done || iter >= iters) {
            cout << "Clustering completed in iteration : " << iter << "\n\n";
            break;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " ms\n";

    // ==========================================================
    // 4. SALVAMENTO DOS DADOS NO DISCO
    // ==========================================================
    string cmd = "mkdir -p " + output_dir;
    if (system(cmd.c_str()) != 0) { /* falha silenciosa permitida apenas para mkdir */ }

    // Salva os pontos
    ofstream pointsFile(output_dir + "/" + to_string(K) + "-points.txt");
    for (int i = 0; i < N; i++) {
        pointsFile << labels[i] << "\n";
    }
    pointsFile.close();

    // Salva os centróides (Mantendo a formatação exata que o seu script Python espera)
    ofstream outfile(output_dir + "/" + to_string(K) + "-clusters.txt");
    outfile << fixed << setprecision(6);
    for (int k = 0; k < K; k++) {
        cout << "Cluster " << k+1 << " centroid : ";
        for (int d = 0; d < dimensions; d++) {
            cout << centroids[k * dimensions + d] << " ";
            outfile << centroids[k * dimensions + d] << " ";
        }
        cout << "\n";
        outfile << "\n";
    }
    outfile.close();

    // Limpa a memória para não vazar
    delete[] points;
    delete[] labels;
    delete[] centroids;
    delete[] sums;
    delete[] counts;

    return 0;
}