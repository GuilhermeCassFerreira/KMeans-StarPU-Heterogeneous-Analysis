#include <omp.h>
#include <cstdio>
int main() {
  printf("omp_get_num_devices() = %d\n", omp_get_num_devices());
  #pragma omp target
  {
    printf("inside target: is_initial_device=%d\n", omp_is_initial_device());
  }
  return 0;
}
