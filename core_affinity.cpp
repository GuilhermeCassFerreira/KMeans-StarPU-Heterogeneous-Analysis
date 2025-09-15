#include <sched.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>

using namespace std;

// Função para detectar os núcleos mais potentes (p-cores)
vector<int> detect_big_cores() {
    cout << "[DEBUG] detect_big_cores called." << endl;
    vector<int> big_cores;
    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    int core_id = 0;

    while (getline(cpuinfo, line)) {
        if (line.find("cpu MHz") != string::npos) {
            size_t pos = line.find(":");
            double freq = stod(line.substr(pos + 1));
            if (freq > 2000) { // Exemplo: considere p-cores com frequência > 2 GHz
                big_cores.push_back(core_id);
            }
            core_id++;
        }
    }
    cpuinfo.close();

    // Exibe os núcleos detectados como p-cores
    if (!big_cores.empty()) {
        cout << "[INFO] Big cores detected (prioritized cores): ";
        for (int core : big_cores) {
            cout << core << " ";
        }
        cout << endl;
    } else {
        cout << "[WARNING] No big cores detected!" << endl;
    }

    return big_cores;
}

// Função para configurar a afinidade para os p-cores
bool set_affinity_to_big_cores(const vector<int>& big_cores) {
    cout << "[DEBUG] set_affinity_to_big_cores called." << endl;

    if (big_cores.empty()) {
        cerr << "[ERROR] No big cores detected! Cannot set affinity to big cores." << endl;
        return false;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int core : big_cores) {
        CPU_SET(core, &mask);
    }

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        perror("[ERROR] sched_setaffinity");
        return false;
    }

    cout << "[INFO] Affinity successfully set to big cores: ";
    for (int core : big_cores) {
        cout << core << " ";
    }
    cout << endl;

    return true;
}

// Função para configurar a afinidade com fallback para todos os núcleos
bool set_affinity_with_fallback(const vector<int>& big_cores) {
    cout << "[DEBUG] set_affinity_with_fallback called." << endl;

    if (set_affinity_to_big_cores(big_cores)) {
        cout << "[INFO] Affinity set to big cores successfully." << endl;
        return true;
    }

    cerr << "[WARNING] Falling back to all cores." << endl;

    // Configura afinidade para todos os núcleos
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < num_cores; i++) {
        CPU_SET(i, &mask);
    }

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        perror("[ERROR] sched_setaffinity");
        return false;
    }

    cout << "[INFO] Affinity set to all cores: ";
    for (int i = 0; i < num_cores; i++) {
        cout << i << " ";
    }
    cout << endl;

    return true;
}