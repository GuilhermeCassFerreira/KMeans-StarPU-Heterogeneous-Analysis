#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

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
        clusterId = 0;
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

    void removeAllPoints() { points.clear(); }
    int getId() { return clusterId; }
    Point getPoint(int pos) { return points[pos]; }
    int getSize() { return points.size(); }
    double getCentroidByPos(int pos) { return centroid[pos]; }
    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }
};

class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;

    void clearClusters() {
        for (int i = 0; i < K; i++)
            clusters[i].removeAllPoints();
    }

    int getNearestClusterId(Point point) { // Encontra o cluster mais próximo de um ponto
        double sum = 0.0, min_dist;
        int NearestClusterId;// ID do cluster mais próximo

        if (dimensions == 1)
            min_dist = abs(clusters[0].getCentroidByPos(0) - point.getVal(0));
            // Distância mínima inicial para 1D
        else {
            for (int i = 0; i < dimensions; i++)
                sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
            min_dist = sqrt(sum);
        }

        NearestClusterId = clusters[0].getId();

        for (int i = 1; i < K; i++) {
            double dist;
            sum = 0.0;

            if (dimensions == 1)
                dist = abs(clusters[i].getCentroidByPos(0) - point.getVal(0));
            else {
                for (int j = 0; j < dimensions; j++)
                    sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
                dist = sqrt(sum);
            }

            if (dist < min_dist) {
                min_dist = dist;
                NearestClusterId = clusters[i].getId();
            }
        }

        return NearestClusterId;
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

            // Atribui pontos ao cluster mais próximo
            for (int i = 0; i < total_points; i++) {
                int currentClusterId = all_points[i].getCluster();
                int nearestClusterId = getNearestClusterId(all_points[i]);
                if (currentClusterId != nearestClusterId) {
                    all_points[i].setCluster(nearestClusterId);
                    done = false;
                }
            }

            clearClusters();

            for (int i = 0; i < total_points; i++)
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);

            // Recalcula centróides
            for (int i = 0; i < K; i++) {
                int ClusterSize = clusters[i].getSize();
                for (int j = 0; j < dimensions; j++) {
                    double sum = 0.0;
                    if (ClusterSize > 0) {
                        for (int p = 0; p < ClusterSize; p++)
                            sum += clusters[i].getPoint(p).getVal(j);
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

        // Salva resultados
        ofstream pointsFile(output_dir + "/" + to_string(K) + "-points.txt");
        for (int i = 0; i < total_points; i++)
            pointsFile << all_points[i].getCluster() << endl;
        pointsFile.close();

        ofstream outfile(output_dir + "/" + to_string(K) + "-clusters.txt");
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
    }
};

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Error: ./kmeans <INPUT> <K> <OUT-DIR>" << endl;
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

    if ((int)all_points.size() < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    int iters = 100;

    auto start = high_resolution_clock::now();
    KMeans kmeans(K, iters, output_dir);
    kmeans.run(all_points);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " ms" << endl;

    return 0;
}
