
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="$ROOT/inputs"
LOG_DIR="$ROOT/logs"
MASTER_LOG="$LOG_DIR/benchmark_master_log.txt"

mkdir -p "$INPUT_DIR" "$LOG_DIR" "$ROOT/results"

# Binaries (ajuste se necessário)
BINS=( "$ROOT/kmeans_antigo" "$ROOT/kmeans" "$ROOT/kmeans_sequencial" )
BIN_NAMES=( "kmeans_antigo" "kmeans" "kmeans_sequencial" )

# Escalonadores a testar (adicionado mais um)
SCHEDS=( "dmda" "ws" "prio" "fifo" )

# repetições por combinação (para kmeans_antigo e kmeans)
REPEATS=${REPEATS:-3}

# timeout em segundos por execução (ajuste se desejar)
TIMEOUT="${TIMEOUT:-86400}"   # 24h por execução

# Cenários de pontos: começa em 50 milhões e aumenta por ADD_POINTS (aditivo)
START_POINTS=${START_POINTS:-50000000}   # 50 milhões
ADD_POINTS=${ADD_POINTS:-80000000}       # aumenta 80 milhões por cenário
NUM_SCENARIOS=${NUM_SCENARIOS:-4}        # 4 cenários

# dims e K que aumentam gradualmente por cenário
START_DIMS=${START_DIMS:-2}
DIMS_INC=${DIMS_INC:-2}

START_K=${START_K:-4}
K_INC=${K_INC:-4}

# Quantos processos usar para gerar inputs (passa para generate_input)
GEN_PROCS="${GEN_PROCS:-16}"

# Header do master log (apenda se não existir)
if [[ ! -f "$MASTER_LOG" ]]; then
  echo "==== RUN START: $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====" >> "$MASTER_LOG"
  echo "scen_index,points,dims,K,scheduler,bin,repeat,rc,exec_time_ms,logfile" >> "$MASTER_LOG"
fi

# função auxiliar para arredondar/inteiro
toint() { awk -v v="$1" 'BEGIN{printf("%.0f", v)}'; }

for ((i=0;i<NUM_SCENARIOS;i++)); do
  scen_index=$((i+1))
  points=$(toint $(( START_POINTS + i * ADD_POINTS )))
  dims=$(( START_DIMS + i * DIMS_INC ))
  K_now=$(( START_K + i * K_INC ))

  infile="$INPUT_DIR/input.txt"   # nome fixo conforme solicitado

  # gera (ou re-gerar) arquivo de entrada: sobrescreve para garantir tamanho correto
  echo "[INFO] Gerando input $infile (points=${points} dims=${dims}) ..." | tee -a "$MASTER_LOG"
  if ! python3 - <<PY
from generate_input import generate_large_input_file
generate_large_input_file(filename=r"$infile", num_points=$points, dimensions=$dims, value_range=(-1000,1000), num_processes=$GEN_PROCS)
PY
  then
    echo "[WARN] Falha/aviso na geração do input (verifique generate_input.py)" | tee -a "$MASTER_LOG"
  fi

  # Executa o binário sequencial APENAS UMA VEZ por cenário (independente do scheduler)
  seq_bin="$ROOT/kmeans_sequencial"
  if [[ -x "$seq_bin" ]]; then
    outdir="$ROOT/results/${points}_d${dims}/sequential/run_1"
    mkdir -p "$outdir"
    perlog="$outdir/run.log"

    echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") | scen=${scen_index} points=${points} dims=${dims} K=${K_now} sched=seq bin=kmeans_sequencial repeat=1 =====" | tee -a "$MASTER_LOG" "$perlog"
    echo "CMD: $seq_bin $infile $K_now $outdir" | tee -a "$MASTER_LOG" "$perlog"

    if command -v timeout >/dev/null 2>&1; then
      timeout "${TIMEOUT}s" "$seq_bin" "$infile" "$K_now" "$outdir" >> "$perlog" 2>&1
      rc=$?
    else
      "$seq_bin" "$infile" "$K_now" "$outdir" >> "$perlog" 2>&1 || rc=$?
      rc=${rc:-0}
    fi

    exec_ms=""
    if grep -q "Execution time:" "$perlog" 2>/dev/null; then
      exec_ms=$(grep "Execution time:" "$perlog" | tail -n1 | sed -E 's/.*Execution time: *([0-9]+) *ms.*/\1/')
    fi

    echo "${scen_index},${points},${dims},${K_now},seq,kmeans_sequencial,1,${rc},${exec_ms},${perlog}" >> "$MASTER_LOG"
    echo "[INFO] finished seq scen=${scen_index} points=${points} bin=kmeans_sequencial rc=${rc} time=${exec_ms:-N/A}ms" | tee -a "$MASTER_LOG"

    echo -e "\n===== BEGIN EXECUTION LOG: scen=${scen_index} points=${points} dims=${dims} K=${K_now} sched=seq bin=kmeans_sequencial run=1 =====" >> "$MASTER_LOG"
    if [[ -f "$perlog" ]]; then
      cat "$perlog" >> "$MASTER_LOG"
    else
      echo "[WARN] per-exec log not found: $perlog" >> "$MASTER_LOG"
    fi
    echo -e "\n===== END EXECUTION LOG =====\n" >> "$MASTER_LOG"

    unset rc
  else
    echo "[WARN] Sequencial binário não encontrado/executável: $seq_bin" | tee -a "$MASTER_LOG"
  fi

  # Para cada escalonador, executar kmeans_antigo e kmeans (cada um REPEATS vezes)
  for sched in "${SCHEDS[@]}"; do
    export STARPU_SCHED="$sched"

    for bin_idx in "${!BINS[@]}"; do
      bname="${BIN_NAMES[$bin_idx]}"
      binpath="${BINS[$bin_idx]}"

      # pula o sequencial aqui (já executado)
      if [[ "$bname" == "kmeans_sequencial" ]]; then
        continue
      fi

      if [[ ! -x "$binpath" ]]; then
        echo "[WARN] Binário não executável/ausente: $binpath" | tee -a "$MASTER_LOG"
        continue
      fi

      for ((r=1; r<=REPEATS; r++)); do
        outdir="$ROOT/results/${points}_d${dims}/${sched}/${bname}/run_${r}"
        mkdir -p "$outdir"
        perlog="$outdir/run.log"

        echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") | scen=${scen_index} points=${points} dims=${dims} K=${K_now} sched=${sched} bin=${bname} repeat=${r} =====" | tee -a "$MASTER_LOG" "$perlog"
        echo "CMD: STARPU_SCHED=${sched} $binpath $infile $K_now $outdir" | tee -a "$MASTER_LOG" "$perlog"

        if command -v timeout >/dev/null 2>&1; then
          timeout "${TIMEOUT}s" "$binpath" "$infile" "$K_now" "$outdir" >> "$perlog" 2>&1
          rc=$?
        else
          "$binpath" "$infile" "$K_now" "$outdir" >> "$perlog" 2>&1 || rc=$?
          rc=${rc:-0}
        fi

        exec_ms=""
        if grep -q "Execution time:" "$perlog" 2>/dev/null; then
          exec_ms=$(grep "Execution time:" "$perlog" | tail -n1 | sed -E 's/.*Execution time: *([0-9]+) *ms.*/\1/')
        fi

        echo "${scen_index},${points},${dims},${K_now},${sched},${bname},${r},${rc},${exec_ms},${perlog}" >> "$MASTER_LOG"
        echo "[INFO] finished scen=${scen_index} points=${points} sched=${sched} bin=${bname} repeat=${r} rc=${rc} time=${exec_ms:-N/A}ms" | tee -a "$MASTER_LOG"

        echo -e "\n===== BEGIN EXECUTION LOG: scen=${scen_index} points=${points} dims=${dims} K=${K_now} sched=${sched} bin=${bname} run=${r} =====" >> "$MASTER_LOG"
        if [[ -f "$perlog" ]]; then
          cat "$perlog" >> "$MASTER_LOG"
        else
          echo "[WARN] per-exec log not found: $perlog" >> "$MASTER_LOG"
        fi
        echo -e "\n===== END EXECUTION LOG =====\n" >> "$MASTER_LOG"

        unset rc
      done
    done
  done
done

echo "==== RUN END: $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====" >> "$MASTER_LOG"
echo "Logs agregados em: $MASTER_LOG