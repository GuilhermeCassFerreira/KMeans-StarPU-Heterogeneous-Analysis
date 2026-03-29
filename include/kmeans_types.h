#ifndef KMEANS_TYPES_H
#define KMEANS_TYPES_H

#include <string>
#include <vector>

class Point {
private:
    int pointId, clusterId;
    int dimensions;
    std::vector<double> values;

    std::vector<double> lineToVec(std::string &line) {
        std::vector<double> values;
        std::string tmp = "";
        for (int i = 0; i < (int)line.length(); i++) {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e') {
                tmp += line[i];
            } else if (tmp.length() > 0) {
                values.push_back(std::stod(tmp));
                tmp = "";
            }
        }
        if (tmp.length() > 0) {
            values.push_back(std::stod(tmp));
            tmp = "";
        }
        return values;
    }

public:
    Point(int id, std::string line) {
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
    void setValues(const std::vector<double>& vals) { values = vals; }
};

class Cluster {
private:
    int clusterId;
    std::vector<double> centroid;
    std::vector<Point> points;

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

    bool removePoint(int pointId) {
        int size = points.size();
        for (int i = 0; i < size; i++) {
            if (points[i].getID() == pointId) {
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

#endif // KMEANS_TYPES_H