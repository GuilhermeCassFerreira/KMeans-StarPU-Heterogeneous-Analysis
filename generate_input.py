import random
import os
import multiprocessing
from multiprocessing import Pool, cpu_count

def generate_points_chunk(args):
    """Gera uma parte dos pontos para o arquivo."""
    filename, start_idx, num_points, dimensions, value_range = args
    data = []
    for _ in range(num_points):
        point = [str(random.uniform(*value_range)) for _ in range(dimensions)]
        data.append(",".join(point) + "\n")
    return data


def generate_large_input_file(filename, num_points, dimensions, value_range, num_processes=None):
    """
    Gera um arquivo massivo com pontos aleatórios usando paralelismo.
    :param filename: Nome do arquivo de saída.
    :param num_points: Número total de pontos.
    :param dimensions: Número de dimensões por ponto.
    :param value_range: Faixa de valores (min, max).
    :param num_processes: Número de processos desejado (opcional).
    """

    # Define número ideal de processos
    available = cpu_count()
    if num_processes is None:
        num_processes = 16
    if num_processes > available:
        num_processes = available
    elif num_processes > 1:
        # Tenta diminuir até um número que o sistema suporte
        while num_processes > available:
            num_processes -= 1

    print(f"🔧 Usando {num_processes} processo(s) de {available} disponível(is)")

    # Divide pontos entre processos
    chunk_size = num_points // num_processes
    args_list = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = num_points if i == num_processes - 1 else (i + 1) * chunk_size
        args_list.append((filename, start_idx, end_idx - start_idx, dimensions, value_range))

    # Executa os processos
    with Pool(num_processes) as pool:
        results = pool.map(generate_points_chunk, args_list)

    # Salva tudo no arquivo
    with open(filename, "w") as f:
        for chunk in results:
            f.writelines(chunk)

    print(f"Arquivo '{filename}' criado com {num_points} pontos.")


if __name__ == "__main__":
    # Exemplo de uso
    generate_large_input_file(
        filename="input.txt",
        num_points=50_000_000,
        dimensions=3,
        value_range=(-1000, 1000),
        num_processes=None  # ou ex: num_processes=4
    )
