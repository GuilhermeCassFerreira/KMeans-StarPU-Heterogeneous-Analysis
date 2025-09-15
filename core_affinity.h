#ifndef CORE_AFFINITY_H
#define CORE_AFFINITY_H

#include <vector>

// Detecta os núcleos mais potentes (p-cores)
std::vector<int> detect_big_cores();

// Configura a afinidade para os p-cores
bool set_affinity_to_big_cores(const std::vector<int>& big_cores);

// Configura a afinidade com fallback para todos os núcleos
bool set_affinity_with_fallback(const std::vector<int>& big_cores);

#endif // CORE_AFFINITY_H