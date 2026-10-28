import random
import sys
import os
import struct
import tempfile
from array import array
from multiprocessing import Pool, cpu_count

# Formato binário (lido por read_points_flat em src/common/io.cpp):
# cabeçalho de 16 bytes ("KMB1" + int32 D + int64 N) seguido de N*D doubles
BIN_MAGIC = b"KMB1"


def write_bin_header(f, num_points, dimensions):
    f.write(BIN_MAGIC + struct.pack("<iq", dimensions, num_points))


def generate_points_chunk(args):
    chunk_idx, num_points, dimensions, value_range, base_seed, tmp_dir, binary = args
    rng = random.Random(base_seed + chunk_idx)
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx:04d}.{'bin' if binary else 'txt'}")
    if binary:
        # Mesma sequência de números do modo texto, gravada como double
        with open(tmp_path, "wb") as f:
            buf = array("d")
            for _ in range(num_points):
                for _ in range(dimensions):
                    buf.append(rng.uniform(*value_range))
                if len(buf) >= 1 << 20:
                    buf.tofile(f)
                    buf = array("d")
            buf.tofile(f)
    else:
        with open(tmp_path, "w") as f:
            for _ in range(num_points):
                point = [str(rng.uniform(*value_range)) for _ in range(dimensions)]
                f.write(",".join(point) + "\n")
    return tmp_path


def generate_large_input_file(filename, num_points, dimensions, value_range, seed=42, num_processes=None):
    available = cpu_count()
    if num_processes is None:
        num_processes = min(16, available)
    else:
        num_processes = min(num_processes, available)

    binary = filename.endswith(".bin")
    print(f"Usando {num_processes} processo(s) de {available} disponivel(is), seed={seed}, "
          f"formato={'binario' if binary else 'texto'}")

    chunk_size = num_points // num_processes
    tmp_dir = tempfile.mkdtemp()

    args_list = []
    for i in range(num_processes):
        count = num_points - i * chunk_size if i == num_processes - 1 else chunk_size
        args_list.append((i, count, dimensions, value_range, seed * 1000, tmp_dir, binary))

    with Pool(num_processes) as pool:
        tmp_files = pool.map(generate_points_chunk, args_list)

    print("Concatenando chunks...")
    with open(filename, "wb" if binary else "w") as out:
        if binary:
            write_bin_header(out, num_points, dimensions)
        for tmp_path in tmp_files:
            with open(tmp_path, "rb" if binary else "r") as f:
                out.write(f.read())
            os.remove(tmp_path)

    os.rmdir(tmp_dir)
    print(f"Arquivo '{filename}' criado com {num_points} pontos.")


def convert_text_to_binary(src, dst):
    # Converte um input.txt existente (um ponto por linha) para o formato binário,
    # preservando exatamente os mesmos valores
    num_points, dimensions = 0, 0
    with open(src, "r") as f, open(dst, "wb") as out:
        write_bin_header(out, 0, 0)  # reescrito no final com N e D
        buf = array("d")
        for line in f:
            values = [float(v) for v in line.replace(",", " ").split()]
            if not values:
                continue
            if dimensions == 0:
                dimensions = len(values)
            buf.extend(values)
            num_points += 1
            if len(buf) >= 1 << 20:
                buf.tofile(out)
                buf = array("d")
        buf.tofile(out)
        out.seek(0)
        write_bin_header(out, num_points, dimensions)
    print(f"Arquivo '{dst}' criado com {num_points} pontos (D={dimensions}).")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        convert_text_to_binary(sys.argv[2], sys.argv[3])
        sys.exit(0)

    num_points = int(sys.argv[1]) if len(sys.argv) > 1 else 250000000
    dimensions = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed       = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    filename   = sys.argv[4]      if len(sys.argv) > 4 else "input.txt"
    num_procs  = int(sys.argv[5]) if len(sys.argv) > 5 else None

    generate_large_input_file(
        filename=filename,
        num_points=num_points,
        dimensions=dimensions,
        value_range=(-1000, 1000),
        seed=seed,
        num_processes=num_procs,
    )
