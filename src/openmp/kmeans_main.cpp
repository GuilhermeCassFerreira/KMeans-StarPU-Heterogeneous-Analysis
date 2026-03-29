#include "kmeans_omp.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

// Assinatura da função que está no io.cpp
bool read_points_from_file(const std::string &filename, std::vector<Point> &all_points, int &N, int &dimensions);

int main(int argc, char **argv) {
    auto start = high_resolution_clock::now();

    if (argc < 3) {
        cout << "Erro de uso. Tente: ./kmeans_seq <INPUT> <K> <OUT-DIR>" << endl;
        return 1;
    }

    string filename = argv[1];
    int K = stoi(argv[2]);
    string output_dir = argv[3];

    vector<Point> all_points;
    int N = 0, dimensions = 0;

    if (!read_points_from_file(filename, all_points, N, dimensions)) {
        return 1;
    }

    KMeansOMP kmeans(K, 100, output_dir, dimensions);
    kmeans.run(all_points, N);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "\nExecution time: " << duration.count() << " ms" << endl;

    return 0;
}