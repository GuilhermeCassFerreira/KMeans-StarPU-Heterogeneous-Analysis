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
#include "../../include/metrics.h"
#include "kmeans_seq.h"

bool read_points_from_file(const std::string &filename, std::vector<Point> &all_points, int &N, int &dimensions);

using namespace std;
using namespace chrono;

static void print_seq_metrics(const SeqMetrics& m) {
    double avg = (m.iter_converged > 0) ? m.t_loop_ms / m.iter_converged : 0.0;
    cout << string(60, '=') << endl;
    cout << "METRICAS FINAIS — Sequencial" << endl;
    cout << string(60, '=') << endl;
    cout << fixed << setprecision(2) << left;
    cout << setw(26) << "Tempo do loop"        << ": " << setw(10) << m.t_loop_ms  << " ms" << endl;
    cout << setw(26) << "Tempo total (c/init)"  << ": " << setw(10) << m.t_total_ms << " ms" << endl;
    cout << setw(26) << "T. medio/iter"         << ": " << setw(10) << avg          << " ms" << endl;
    cout << setw(26) << "Iteracoes"             << ": " << m.iter_converged << " / max " << m.iter_max << endl;
    cout << setprecision(4);
    cout << setw(26) << "SSE"                   << ": " << scientific << m.sse << endl;
    cout << string(60, '=') << endl;
    cout << defaultfloat;
}

int main(int argc, char **argv) {
    auto t_prog_start = high_resolution_clock::now();

    if (argc < 4 || argc > 6) {
        cout << "Uso: ./kmeans_seq <INPUT> <K> <OUT-DIR> [SEED] [NITERS]" << endl;
        return 1;
    }

    string filename  = argv[1];
    int K            = stoi(argv[2]);
    string output_dir = argv[3];
    int seed         = (argc >= 5) ? stoi(argv[4]) : 42;
    int nIters       = (argc >= 6) ? stoi(argv[5]) : 100;

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

    double *points    = new double[N * dimensions];
    int    *labels    = new int[N];
    double *centroids = new double[K * dimensions];
    double *sums      = new double[K * dimensions];
    int    *counts    = new int[K];

    for (int i = 0; i < N; i++)
        for (int d = 0; d < dimensions; d++)
            points[i * dimensions + d] = all_points[i].getVal(d);

    srand(seed);
    cout << ">> Inicializando centroides com SEED: " << seed << endl;
    vector<int> chosen;
    while ((int)chosen.size() < K) {
        int r = rand() % N;
        if (find(chosen.begin(), chosen.end(), r) == chosen.end())
            chosen.push_back(r);
    }
    for (int i = 0; i < K; ++i)
        for (int d = 0; d < dimensions; d++)
            centroids[i * dimensions + d] = points[chosen[i] * dimensions + d];

    auto t_loop_start = high_resolution_clock::now();
    int iter_converged = nIters;

    for (int iter = 1; iter <= nIters; ++iter) {
        memset(sums,   0, K * dimensions * sizeof(double));
        memset(counts, 0, K * sizeof(int));

        int changes = assign_point_to_cluster_seq(points, centroids, labels, N, K, dimensions);
        calculate_partial_sums_seq(points, labels, sums, counts, N, K, dimensions);
        update_centroids_seq(sums, counts, centroids, K, dimensions);

        cout << "[SEQ] Iteracao " << iter << " | mudancas: " << changes << endl;

        if (changes == 0) {
            iter_converged = iter;
            cout << "[SEQ] Convergiu na iteracao " << iter << endl;
            break;
        }
    }

    auto t_loop_end = high_resolution_clock::now();

    // SSE
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        int c = labels[i] - 1;
        if (c < 0 || c >= K) continue;
        for (int d = 0; d < dimensions; d++) {
            double diff = points[i * dimensions + d] - centroids[c * dimensions + d];
            sse += diff * diff;
        }
    }

    // Salvar resultados
    string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());

    ofstream pointsFile(output_dir + "/" + to_string(K) + "-points.txt");
    for (int i = 0; i < N; i++) pointsFile << labels[i] << "\n";
    pointsFile.close();

    ofstream clustersFile(output_dir + "/" + to_string(K) + "-clusters.txt");
    clustersFile << fixed << setprecision(6);
    for (int k = 0; k < K; k++) {
        for (int d = 0; d < dimensions; d++)
            clustersFile << centroids[k * dimensions + d] << " ";
        clustersFile << "\n";
    }
    clustersFile.close();
    cout << "[INFO] Arquivos salvos em: " << output_dir << endl;

    auto t_prog_end = high_resolution_clock::now();

    SeqMetrics m{};
    m.t_loop_ms      = duration<double, milli>(t_loop_end  - t_loop_start).count();
    m.t_total_ms     = duration<double, milli>(t_prog_end  - t_prog_start).count();
    m.iter_converged = iter_converged;
    m.iter_max       = nIters;
    m.sse            = sse;
    print_seq_metrics(m);

    delete[] points; delete[] labels; delete[] centroids;
    delete[] sums;   delete[] counts;
    return 0;
}
