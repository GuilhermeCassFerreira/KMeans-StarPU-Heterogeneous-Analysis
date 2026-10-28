#include "../../include/kmeans_types.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>

bool read_points_from_file(const std::string &filename, std::vector<Point> &all_points, int &N, int &dimensions) {
    std::ifstream infile(filename.c_str());// Open the file
    if (!infile.is_open()) {
        std::cout << "Error: Failed to open file: " << filename << std::endl;
        return false;
    }

    int pointId = 1;
    std::string line;
    while (std::getline(infile, line)) {// enquanto houver linhas para ler
        Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }
    infile.close();

    N = all_points.size();
    if (N > 0) {
        dimensions = all_points[0].getDimensions();
    }

    return true;
}

// Formato binário: cabeçalho de 16 bytes ("KMB1" + int32 D + int64 N),
// seguido de N*D doubles, ponto a ponto (mesmo layout do array dos kernels)
static const char BIN_MAGIC[4] = {'K', 'M', 'B', '1'};

static bool read_binary_points(FILE *f, const std::string &filename, std::vector<double> &values, int &N, int &dimensions) {
    int32_t d32 = 0;
    int64_t n64 = 0;
    if (fread(&d32, sizeof(d32), 1, f) != 1 || fread(&n64, sizeof(n64), 1, f) != 1 || d32 <= 0 || n64 <= 0) {
        std::cout << "Error: Invalid binary header in file: " << filename << std::endl;
        return false;
    }
    N = (int)n64;
    dimensions = d32;
    values.resize((size_t)N * dimensions);

    // Leitura em blocos de até 1 GB
    const size_t block = (size_t)1 << 27;
    size_t total = values.size(), done = 0;
    while (done < total) {
        size_t want = std::min(block, total - done);
        size_t got = fread(values.data() + done, sizeof(double), want, f);
        if (got != want) {
            std::cout << "Error: Truncated binary file: " << filename << std::endl;
            return false;
        }
        done += got;
    }
    return true;
}

static bool read_text_points(std::ifstream &infile, std::vector<double> &values, int &N, int &dimensions) {
    std::string line;
    dimensions = 0;
    N = 0;
    while (std::getline(infile, line)) {
        const char *p = line.c_str();
        char *end = nullptr;
        int count = 0;
        while (*p) {
            // Pula separadores (vírgula, espaço) até o próximo número
            if (!((*p >= '0' && *p <= '9') || *p == '.' || *p == '+' || *p == '-')) { p++; continue; }
            double v = strtod(p, &end);
            if (end == p) { p++; continue; }
            values.push_back(v);
            count++;
            p = end;
        }
        if (count == 0) continue;
        if (dimensions == 0) dimensions = count;
        N++;
    }
    return N > 0;
}

// Lê os pontos direto para um array contíguo (N*D doubles), sem objetos Point.
// Detecta o formato pelo cabeçalho: binário ("KMB1") ou texto (um ponto por linha).
bool read_points_flat(const std::string &filename, std::vector<double> &values, int &N, int &dimensions) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) {
        std::cout << "Error: Failed to open file: " << filename << std::endl;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    bool is_binary = fread(magic, 1, 4, f) == 4 && memcmp(magic, BIN_MAGIC, 4) == 0;
    if (is_binary) {
        bool ok = read_binary_points(f, filename, values, N, dimensions);
        fclose(f);
        return ok;
    }
    fclose(f);

    std::ifstream infile(filename.c_str());
    return read_text_points(infile, values, N, dimensions);
}