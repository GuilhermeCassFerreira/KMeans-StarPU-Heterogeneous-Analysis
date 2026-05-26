#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include "../../include/metrics.h"

using namespace std;

/* ========================================================================== */
/* Helpers internos                                                           */
/* ========================================================================== */

static double pct(long part, long total) {
    if (total <= 0) return 0.0;
    return (double)part * 100.0 / (double)total;
}

static string mode_name(int mode) {
    if (mode == 1) return "FULL GPU";
    if (mode == 2) return "HIBRIDO (CPU+GPU)";
    return "FULL CPU";
}

static void print_sep() {
    cout << string(60, '=') << endl;
}

/* ========================================================================== */
/* OpenMP/MPI                                                                 */
/* ========================================================================== */

void print_omp_metrics(int rank, int world_size, const OmpMetrics& m) {
    /* --- Gather per-rank data (all ranks must participate) --- */
    double my_loop  = m.t_loop_ms;
    double my_total = m.t_total_ms;
    /* pack task counts: [total_assign, gpu_assign, total_calc, gpu_calc, total_update, gpu_update] */
    long my_counts[6] = { m.total_assign,    m.gpu_assign,
                           m.total_calculate, m.gpu_calculate,
                           m.total_update,    m.gpu_update };

    vector<double> all_loop(world_size), all_total(world_size);
    vector<long>   all_counts(world_size * 6);

    MPI_Gather(&my_loop,   1, MPI_DOUBLE, all_loop.data(),   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_total,  1, MPI_DOUBLE, all_total.data(),  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(my_counts,  6, MPI_LONG,   all_counts.data(), 6, MPI_LONG,   0, MPI_COMM_WORLD);

    if (rank != 0) return;

    /* --- Aggregate --- */
    double t_loop_min  = all_loop[0],  t_loop_max  = all_loop[0];
    double t_total_min = all_total[0], t_total_max = all_total[0];
    long g_assign=0, g_gpu_assign=0, g_calc=0, g_gpu_calc=0, g_update=0, g_gpu_update=0;

    for (int i = 0; i < world_size; i++) {
        t_loop_min  = min(t_loop_min,  all_loop[i]);
        t_loop_max  = max(t_loop_max,  all_loop[i]);
        t_total_min = min(t_total_min, all_total[i]);
        t_total_max = max(t_total_max, all_total[i]);
        g_assign      += all_counts[i*6+0];
        g_gpu_assign  += all_counts[i*6+1];
        g_calc        += all_counts[i*6+2];
        g_gpu_calc    += all_counts[i*6+3];
        g_update      += all_counts[i*6+4];
        g_gpu_update  += all_counts[i*6+5];
    }
    long cpu_assign    = g_assign  - g_gpu_assign;
    long cpu_calculate = g_calc    - g_gpu_calc;
    long cpu_update    = g_update  - g_gpu_update;

    /* avg/iter uses the slowest rank's loop time */
    double avg = (m.iter_converged > 0) ? t_loop_max / m.iter_converged : 0.0;

    print_sep();
    cout << "METRICAS FINAIS — OpenMP/MPI" << endl;
    print_sep();
    cout << fixed << setprecision(2) << left;
    cout << setw(26) << "Versao"       << ": " << mode_name(m.mode) << endl;
    cout << setw(26) << "Configuracao" << ": "
         << m.mpi_ranks << " rank(s) MPI x " << m.omp_threads << " threads OMP"
         << "  |  chunks: " << m.num_chunks << endl;
    cout << endl;

    if (world_size == 1) {
        cout << setw(26) << "Tempo do loop"       << ": " << setw(10) << t_loop_max  << " ms" << endl;
        cout << setw(26) << "Tempo total (c/init)" << ": " << setw(10) << t_total_max << " ms" << endl;
    } else {
        cout << setw(26) << "Tempo do loop (max)"       << ": " << setw(10) << t_loop_max  << " ms"
             << "  [min: " << fixed << setprecision(2) << t_loop_min  << "]" << endl;
        cout << setw(26) << "Tempo total (max, c/init)" << ": " << setw(10) << t_total_max << " ms"
             << "  [min: " << t_total_min << "]" << endl;
    }
    cout << setw(26) << "T. medio/iter"       << ": " << setw(10) << avg << " ms" << endl;
    cout << setw(26) << "Iteracoes"           << ": " << m.iter_converged << " / max " << m.iter_max << endl;
    cout << setprecision(4);
    cout << setw(26) << "SSE"                 << ": " << scientific << m.sse << defaultfloat << endl;

    if (world_size > 1) {
        cout << endl << "Tempos por rank:" << endl;
        cout << fixed << setprecision(2);
        for (int i = 0; i < world_size; i++) {
            double rank_avg = (m.iter_converged > 0) ? all_loop[i] / m.iter_converged : 0.0;
            cout << "  Rank " << i
                 << ": loop=" << setw(10) << all_loop[i]  << " ms"
                 << "  avg/iter=" << setw(8) << rank_avg   << " ms"
                 << "  total="    << setw(10) << all_total[i] << " ms" << endl;
        }
    }

    cout << endl;
    cout << "Dispositivos utilizados (por operacao — global):" << endl;
    cout << fixed << setprecision(1);
    cout << "  " << left << setw(14) << "Operacao"
         << setw(24) << "CPU (N / %)"
         << setw(24) << "GPU (N / %)" << endl;
    cout << "  " << string(60, '-') << endl;

    auto row = [&](const string& name, long cpu, long gpu) {
        long tot = cpu + gpu;
        ostringstream c, g;
        c << cpu << " (" << pct(cpu, tot) << "%)";
        g << gpu << " (" << pct(gpu, tot) << "%)";
        cout << "  " << left << setw(14) << name
             << setw(24) << c.str()
             << setw(24) << g.str() << endl;
    };

    row("Assign",    cpu_assign,    g_gpu_assign);
    row("Calculate", cpu_calculate, g_gpu_calc);
    row("Update",    cpu_update,    g_gpu_update);

    print_sep();
    cout << defaultfloat;
}

/* ========================================================================== */
/* StarPU/MPI                                                                 */
/* ========================================================================== */

void print_starpu_metrics(int rank, int world_size, const StarPUMetrics& m) {
    /* --- Gather per-rank data (all ranks must participate) --- */
    double my_loop  = m.t_loop_ms;
    double my_total = m.t_total_ms;
    /* pack cpu counts: [assign, calculate, clean, update, accumulate] */
    long my_cpu[5]  = { m.cpu_assign,  m.cpu_calculate,  m.cpu_clean,  m.cpu_update,  m.cpu_accumulate  };
    long my_cuda[5] = { m.cuda_assign, m.cuda_calculate, m.cuda_clean, m.cuda_update, m.cuda_accumulate };

    vector<double> all_loop(world_size), all_total(world_size);
    vector<long>   all_cpu(world_size * 5), all_cuda(world_size * 5);

    MPI_Gather(&my_loop,  1, MPI_DOUBLE, all_loop.data(),  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_total, 1, MPI_DOUBLE, all_total.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(my_cpu,    5, MPI_LONG,   all_cpu.data(),   5, MPI_LONG,   0, MPI_COMM_WORLD);
    MPI_Gather(my_cuda,   5, MPI_LONG,   all_cuda.data(),  5, MPI_LONG,   0, MPI_COMM_WORLD);

    if (rank != 0) return;

    /* --- Aggregate --- */
    double t_loop_min  = all_loop[0],  t_loop_max  = all_loop[0];
    double t_total_min = all_total[0], t_total_max = all_total[0];

    long g_cpu[5]  = {}, g_cuda[5] = {};
    for (int i = 0; i < world_size; i++) {
        t_loop_min  = min(t_loop_min,  all_loop[i]);
        t_loop_max  = max(t_loop_max,  all_loop[i]);
        t_total_min = min(t_total_min, all_total[i]);
        t_total_max = max(t_total_max, all_total[i]);
        for (int k = 0; k < 5; k++) {
            g_cpu[k]  += all_cpu[i*5+k];
            g_cuda[k] += all_cuda[i*5+k];
        }
    }

    long total_cpu  = g_cpu[0]  + g_cpu[1]  + g_cpu[2]  + g_cpu[3]  + g_cpu[4];
    long total_cuda = g_cuda[0] + g_cuda[1] + g_cuda[2] + g_cuda[3] + g_cuda[4];

    double avg = (m.iter_converged > 0) ? t_loop_max / m.iter_converged : 0.0;

    print_sep();
    cout << "METRICAS FINAIS — StarPU/MPI" << endl;
    print_sep();
    cout << fixed << setprecision(2) << left;
    cout << setw(26) << "Workers (rank 0)"    << ": "
         << m.ncpu_workers << " CPU  +  " << m.ncuda_workers << " GPU CUDA" << endl;
    cout << setw(26) << "Ranks MPI"           << ": " << m.mpi_ranks << endl;
    cout << endl;

    if (world_size == 1) {
        cout << setw(26) << "Tempo do loop"       << ": " << setw(10) << t_loop_max  << " ms" << endl;
        cout << setw(26) << "Tempo total (c/init)" << ": " << setw(10) << t_total_max << " ms" << endl;
    } else {
        cout << setw(26) << "Tempo do loop (max)"       << ": " << setw(10) << t_loop_max  << " ms"
             << "  [min: " << fixed << setprecision(2) << t_loop_min  << "]" << endl;
        cout << setw(26) << "Tempo total (max, c/init)" << ": " << setw(10) << t_total_max << " ms"
             << "  [min: " << t_total_min << "]" << endl;
    }
    cout << setw(26) << "T. medio/iter"       << ": " << setw(10) << avg << " ms" << endl;
    cout << setw(26) << "Iteracoes"           << ": " << m.iter_converged << " / max " << m.iter_max << endl;
    cout << setprecision(4);
    cout << setw(26) << "SSE"                 << ": " << scientific << m.sse << defaultfloat << endl;

    if (world_size > 1) {
        cout << endl << "Tempos por rank:" << endl;
        cout << fixed << setprecision(2);
        for (int i = 0; i < world_size; i++) {
            double rank_avg = (m.iter_converged > 0) ? all_loop[i] / m.iter_converged : 0.0;
            cout << "  Rank " << i
                 << ": loop=" << setw(10) << all_loop[i]  << " ms"
                 << "  avg/iter=" << setw(8) << rank_avg   << " ms"
                 << "  total="    << setw(10) << all_total[i] << " ms" << endl;
        }
    }

    cout << endl;
    cout << "Distribuicao por kernel (CPU vs CUDA) — global:" << endl;
    cout << fixed << setprecision(1);
    cout << "  " << left << setw(14) << "Kernel"
         << setw(24) << "CPU (N / %)"
         << setw(24) << "GPU CUDA (N / %)" << endl;
    cout << "  " << string(60, '-') << endl;

    auto row = [&](const string& name, long cpu, long gpu) {
        long tot = cpu + gpu;
        ostringstream c, g;
        c << cpu << " (" << pct(cpu, tot) << "%)";
        g << gpu << " (" << pct(gpu, tot) << "%)";
        cout << "  " << left << setw(14) << name
             << setw(24) << c.str()
             << setw(24) << g.str() << endl;
    };

    row("Assign",     g_cpu[0], g_cuda[0]);
    row("Calculate",  g_cpu[1], g_cuda[1]);
    row("Clean",      g_cpu[2], g_cuda[2]);
    row("Update",     g_cpu[3], g_cuda[3]);
    row("Accumulate", g_cpu[4], g_cuda[4]);
    cout << "  " << string(60, '-') << endl;
    row("TOTAL",      total_cpu, total_cuda);

    if (world_size > 1) {
        cout << endl << "Tarefas por rank MPI:" << endl;
        for (int i = 0; i < world_size; i++) {
            long rank_cpu  = 0, rank_cuda = 0;
            for (int k = 0; k < 5; k++) { rank_cpu += all_cpu[i*5+k]; rank_cuda += all_cuda[i*5+k]; }
            long grand = rank_cpu + rank_cuda;
            cout << "  Rank " << i << ": "
                 << rank_cpu  << " CPU + " << rank_cuda << " CUDA = "
                 << grand << " total (" << pct(grand, total_cpu + total_cuda) << "%)" << endl;
        }
    }

    print_sep();
    cout << defaultfloat;
}
