import random
import os

def generate_large_input_file(filename, num_points, dimensions, value_range):
    """
    Gera um arquivo de entrada massivo com pontos aleatórios.

    :param filename: Nome do arquivo de saída.
    :param num_points: Número de pontos a serem gerados.
    :param dimensions: Número de dimensões de cada ponto.
    :param value_range: Intervalo de valores (mínimo, máximo).
    """
    with open(filename, "w") as f:
        for _ in range(num_points):
            point = [round(random.uniform(*value_range), 2) for _ in range(dimensions)]
            f.write(" ".join(map(str, point)) + "\n")

# Configurações
output_file = "massive_input.txt"  # Nome do arquivo de saída
num_points = 5000000              # Número de pontos (50 milhões, aumente conforme necessário)
dimensions = 2                     # Número de dimensões (exemplo: 2D)
value_range = (-1000, 1000)        # Intervalo de valores (mínimo, máximo)

# Remove o arquivo se já existir
if os.path.exists(output_file):
    os.remove(output_file)

# Gera o arquivo
generate_large_input_file(output_file, num_points, dimensions, value_range)
print(f"Arquivo '{output_file}' gerado com {num_points} pontos.")