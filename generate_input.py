import random
import sys
import multiprocessing
from multiprocessing import Pool, cpu_count

def generate_points_chunk(args):
    chunk_idx, num_points, dimensions, value_range, base_seed = args
    rng = random.Random(base_seed + chunk_idx)
    data = []
    for _ in range(num_points):
        point = [str(rng.uniform(*value_range)) for _ in range(dimensions)]
        data.append(",".join(point) + "\n")
    return data


def generate_large_input_file(filename, num_points, dimensions, value_range, seed=42, num_processes=None):
    available = cpu_count()
    if num_processes is None:
        num_processes = min(16, available)
    else:
        num_processes = min(num_processes, available)

    print(f"Usando {num_processes} processo(s) de {available} disponivel(is), seed={seed}")

    chunk_size = num_points // num_processes
    args_list = []
    for i in range(num_processes):
        start = i * chunk_size
        count = num_points - start if i == num_processes - 1 else chunk_size
        args_list.append((i, count, dimensions, value_range, seed * 1000))

    with Pool(num_processes) as pool:
        results = pool.map(generate_points_chunk, args_list)

    with open(filename, "w") as f:
        for chunk in results:
            f.writelines(chunk)

    print(f"Arquivo '{filename}' criado com {num_points} pontos.")


if __name__ == "__main__":
    num_points = int(sys.argv[1]) if len(sys.argv) > 1 else 250000000
    dimensions = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed       = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    filename   = sys.argv[4]      if len(sys.argv) > 4 else "input.txt"

    generate_large_input_file(
        filename=filename,
        num_points=num_points,
        dimensions=dimensions,
        value_range=(-1000, 1000),
        seed=seed,
    )
