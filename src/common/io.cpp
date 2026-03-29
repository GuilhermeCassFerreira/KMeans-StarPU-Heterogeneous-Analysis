#include "../../include/kmeans_types.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

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