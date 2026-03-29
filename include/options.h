#ifndef OPTIONS_H
#define OPTIONS_H

#include <string>

struct KMeansOptions {
    int K;
    int iters;
    int chunk_size;
    int dimensions;
    int total_points;
    std::string input_file;
    std::string output_dir;
    bool use_heterogeneous_chunks;
    int mpi_rank;
    int world_size;
};

#endif // OPTIONS_H