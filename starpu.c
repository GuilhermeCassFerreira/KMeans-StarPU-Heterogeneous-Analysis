#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <starpu.h>
#include <time.h>

#define N 9000000000ULL  // 1 bilhão de elementos (~4GB RAM)
#define NUM_TASKS 16     // Quantidade de tasks para paralelizar

// Task que soma uma parte do vetor
void soma_parcial(void *buffers[], void *cl_arg) {
    int *vetor = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
    int *resultado = (int *)STARPU_VARIABLE_GET_PTR(buffers[1]);

    size_t n = *(size_t *)cl_arg;
    int soma = 0;

    for (size_t i = 0; i < n; i++) {
        soma += vetor[i];
    }

    *resultado = soma;
}

// Codelet para a task soma_parcial
static struct starpu_codelet cl_soma_parcial = {
    .cpu_funcs = {soma_parcial},
    .nbuffers = 2,
    .modes = {STARPU_R, STARPU_W},
    .name = "soma_parcial"
};

int main() {
    clock_t inicio = clock();

    starpu_init(NULL);

    // Aloca vetor grande
    int *vetor = malloc(N * sizeof(int));
    if (!vetor) {
        fprintf(stderr, "Erro ao alocar vetor\n");
        return 1;
    }

    // Inicializa vetor com 1 para facilitar conferência
    for (size_t i = 0; i < N; i++) {
        vetor[i] = 1;
    }

    size_t bloco_tamanho = N / NUM_TASKS;

    // Vetor para resultados parciais
    int *resultados = malloc(NUM_TASKS * sizeof(int));
    if (!resultados) {
        fprintf(stderr, "Erro ao alocar resultados\n");
        free(vetor);
        return 1;
    }

    starpu_data_handle_t handles_vetor[NUM_TASKS];
    starpu_data_handle_t handles_resultado[NUM_TASKS];

    // Criar e submeter tasks
    for (int i = 0; i < NUM_TASKS; i++) {
        size_t offset = i * bloco_tamanho;
        size_t tamanho_atual = (i == NUM_TASKS - 1) ? (N - offset) : bloco_tamanho;

        starpu_vector_data_register(&handles_vetor[i], STARPU_MAIN_RAM, (uintptr_t)(vetor + offset), tamanho_atual, sizeof(int));
        starpu_variable_data_register(&handles_resultado[i], STARPU_MAIN_RAM, (uintptr_t)&resultados[i], sizeof(int));

        struct starpu_task *task = starpu_task_create();
        task->cl = &cl_soma_parcial;
        task->handles[0] = handles_vetor[i];
        task->handles[1] = handles_resultado[i];
        task->cl_arg = &tamanho_atual;
        task->cl_arg_size = sizeof(size_t);

        starpu_task_submit(task);
    }

    starpu_task_wait_for_all();

    // Soma final dos resultados parciais
    long long soma_total = 0;
    for (int i = 0; i < NUM_TASKS; i++) {
        soma_total += resultados[i];
        starpu_data_unregister(handles_vetor[i]);
        starpu_data_unregister(handles_resultado[i]);
    }

    printf("Soma total: %lld\n", soma_total);

    starpu_shutdown();

    clock_t fim = clock();
    double tempo_total = (double)(fim - inicio) / CLOCKS_PER_SEC;
    printf("Tempo total de execucao: %.6f segundos\n", tempo_total);

    free(vetor);
    free(resultados);

    return 0;
}
