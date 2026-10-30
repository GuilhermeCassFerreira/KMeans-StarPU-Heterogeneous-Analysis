"""Fração dos pontos na GPU (gpu_ratio) do OpenMP híbrido, para CPU e GPU terminarem juntas.

Tempo de um dispositivo para processar um ponto numa iteração:
  t = K·D / velocidade + (8·D + 4) / banda
O 1º termo é a atribuição (K·D elementos de conta, limitada pela velocidade de cálculo); o 2º é
a acumulação (lê o ponto e o rótulo, limitada pela banda de memória), como no modelo Roofline
(Williams, Waterman e Patterson, CACM 2009). A velocidade é o teto que o código atinge: 1
elemento por ciclo por núcleo na CPU e por SM na GPU. Os pontos são divididos na proporção da
velocidade de cada dispositivo (Beaumont et al., IEEE TPDS 2001):
  r_gpu = t_cpu / (t_cpu + t_gpu)

Padrão: AWS g6.4xlarge (GPU L4: 58 SMs a 2,04 GHz, 300 GB/s; CPU AMD EPYC 7R13 a 3,6 GHz, com
6 threads de cálculo, já que a 7ª só controla a GPU).
Uso: python3 scripts/gpu_ratio.py [--cpu-threads 6] [--cpu-ghz 3.6] [--cpu-bw 50] ...
"""
import argparse

SCENARIOS = [("PQ", 256, 16), ("Deep", 256, 96), ("SIFT", 256, 128)]


def ratio(k, d, a):
    t_gpu = k * d / (a.gpu_sms * a.gpu_ghz * 1e9) + (8 * d + 4) / (a.gpu_bw * 1e9)
    t_cpu = k * d / (a.cpu_threads * a.cpu_ghz * 1e9) + (8 * d + 4) / (a.cpu_bw * 1e9)
    return t_cpu / (t_cpu + t_gpu)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cpu-threads", type=int, default=6, help="threads de CPU que calculam")
    p.add_argument("--cpu-ghz", type=float, default=3.6)
    p.add_argument("--cpu-bw", type=float, default=50.0, help="banda de memória da CPU (GB/s)")
    p.add_argument("--gpu-sms", type=int, default=58)
    p.add_argument("--gpu-ghz", type=float, default=2.04)
    p.add_argument("--gpu-bw", type=float, default=300.0, help="banda de memória da GPU (GB/s)")
    a = p.parse_args()
    for name, k, d in SCENARIOS:
        print(f"{name:5s} K={k} D={d:3d}  gpu_ratio = {ratio(k, d, a):.3f}")


if __name__ == "__main__":
    main()
