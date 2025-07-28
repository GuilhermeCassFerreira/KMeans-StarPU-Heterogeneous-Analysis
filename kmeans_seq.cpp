#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
// #include <omp.h> // Comente ou remova esta linha se quiser garantir que não haja OpenMP

using namespace std;
using namespace chrono;

class Point
{
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

    vector<double> lineToVec(string &line)
    {
        vector<double> values;
        string tmp = "";

        for (int i = 0; i < (int)line.length(); i++)
        {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e')
            {
                tmp += line[i];
            }
            else if (tmp.length() > 0)
            {
                values.push_back(stod(tmp));
                tmp = "";
            }
        }
        if (tmp.length() > 0)
        {
            values.push_back(stod(tmp));
            tmp = "";
        }

        return values;
    }

public:
    Point(int id, string line)
    {
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

class Cluster
{
private:
    int clusterId;
    vector<double> centroid;
    vector<Point> points;

public:
    Cluster(int clusterId, Point centroid)
    {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid.getDimensions(); i++)
        {
            this->centroid.push_back(centroid.getVal(i));
        }
        this->addPoint(centroid);
    }

    void addPoint(Point p)
    {
        p.setCluster(this->clusterId);
        points.push_back(p);
    }

    bool removePoint(int pointId)
    {
        int size = points.size();

        for (int i = 0; i < size; i++)
        {
            if (points[i].getID() == pointId)
            {
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

class KMeans
{
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;

    void clearClusters()
    {
        for (int i = 0; i < K; i++)
        {
            clusters[i].removeAllPoints();
        }
    }

    int getNearestClusterId(Point point)
    {
        double sum = 0.0, min_dist;
        int NearestClusterId;
        if (dimensions == 1)
        {
            min_dist = abs(clusters[0].getCentroidByPos(0) - point.getVal(0));
        }
        else
        {
            for (int i = 0; i < dimensions; i++)
            {
                sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
            }
            min_dist = sqrt(sum);
        }
        NearestClusterId = clusters[0].getId();

        for (int i = 1; i < K; i++)
        {
            double dist;
            sum = 0.0;

            if (dimensions == 1)
            {
                dist = abs(clusters[i].getCentroidByPos(0) - point.getVal(0));
            }
            else
            {
                for (int j = 0; j < dimensions; j++)
                {
                    sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
                }

                dist = sqrt(sum);
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                NearestClusterId = clusters[i].getId();
            }
        }

        return NearestClusterId;
    }

public:
    KMeans(int K, int iterations, string output_dir)
    {
        this->K = K;
        this->iters = iterations;
        this->output_dir = output_dir;
    }

    void run(vector<Point> &all_points)
    {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        // Initializing Clusters
        vector<int> used_pointIds;

        for (int i = 1; i <= K; i++)
        {
            while (true)
            {
                int index = rand() % total_points;

                if (find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end())
                {
                    used_pointIds.push_back(index);
                    all_points[index].setCluster(i);
                    Cluster cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }
        cout << "Clusters initialized = " << clusters.size() << endl
             << endl;

        cout << "Running K-Means Clustering.." << endl;

        int iter = 1;
        while (true)
        {
            cout << "Iter - " << iter << "/" << iters << endl;
            bool done = true;

            // Add all points to their nearest cluster
            // Para garantir que esta seção seja sequencial, compile sem -fopenmp
            // ou defina OMP_NUM_THREADS=1
            // #pragma omp parallel for reduction(&&: done) 
            for (int i = 0; i < total_points; i++)
            {
                int currentClusterId = all_points[i].getCluster();
                int nearestClusterId = getNearestClusterId(all_points[i]);

                if (currentClusterId != nearestClusterId)
                {
                    all_points[i].setCluster(nearestClusterId);
                    done = false;
                }
            }

            // clear all existing clusters
            clearClusters();

            // reassign points to their new clusters
            for (int i = 0; i < total_points; i++)
            {
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            // Recalculating the center of each cluster
            for (int i = 0; i < K; i++)
            {
                int ClusterSize = clusters[i].getSize();

                for (int j = 0; j < dimensions; j++)
                {
                    double sum = 0.0;
                    if (ClusterSize > 0)
                    {
                        // Para garantir que esta seção seja sequencial, compile sem -fopenmp
                        // ou defina OMP_NUM_THREADS=1
                        // #pragma omp parallel for reduction(+: sum) 
                        for (int p = 0; p < ClusterSize; p++)
                        {
                            sum += clusters[i].getPoint(p).getVal(j);
                        }
                        clusters[i].setCentroidByPos(j, sum / ClusterSize);
                    }
                }
            }

            if (done || iter >= iters)
            {
                cout << "Clustering completed in iteration : " << iter << endl
                     << endl;
                break;
            }
            iter++;
        }

        ofstream pointsFile;
        pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);

        for (int i = 0; i < total_points; i++)
        {
            pointsFile << all_points[i].getCluster() << endl;
        }

        pointsFile.close();

        // Write cluster centers to file
        ofstream outfile;
        outfile.open(output_dir + "/" + to_string(K) + "-clusters.txt");
        if (outfile.is_open())
        {
            for (int i = 0; i < K; i++)
            {
                cout << "Cluster " << clusters[i].getId() << " centroid : ";
                for (int j = 0; j < dimensions; j++)
                {
                    cout << clusters[i].getCentroidByPos(j) << " ";    // Output to console
                    outfile << clusters[i].getCentroidByPos(j) << " "; // Output to file
                }
                cout << endl;
                outfile << endl;
            }
            outfile.close();
        }
        else
        {
            cout << "Error: Unable to write to clusters.txt";
        }
    }
};

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cout << "Error: command-line argument count mismatch. \n ./kmeans <INPUT> <K> <OUT-DIR>" << endl;
        return 1;
    }

    string output_dir = argv[3];
    int K = atoi(argv[2]);
    string filename = argv[1];
    ifstream infile(filename.c_str());

    if (!infile.is_open())
    {
        cout << "Error: Failed to open file." << endl;
        return 1;
    }

    int pointId = 1;
    vector<Point> all_points;
    string line;

    while (getline(infile, line))
    {
        Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }

    infile.close();
    cout << "\nData fetched successfully!" << endl
         << endl;

    if ((int)all_points.size() < K)
    {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    int iters = 100;

    // Início da medição de tempo
    auto start = high_resolution_clock::now();

    KMeans kmeans(K, iters, output_dir);
    kmeans.run(all_points);

    // Fim da medição de tempo
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Execution time: " << duration.count() << " ms" << endl;

    return 0;
}
