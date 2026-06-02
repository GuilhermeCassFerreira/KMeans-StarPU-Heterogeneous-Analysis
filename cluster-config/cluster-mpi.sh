#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${SCRIPT_DIR}/cluster-mpi.env"
STATE_DIR="${SCRIPT_DIR}/.cluster"
CONTROL_IPS_FILE="${STATE_DIR}/control_ips.txt"
MPI_IPS_FILE="${STATE_DIR}/mpi_ips.txt"
HOSTFILE_PATH="${STATE_DIR}/hostfile"
LOCAL_TRACES_DIR="${REPO_ROOT}/traces"

SSH_USER="ec2-user"
SSH_PORT=22
SSH_KEY="${HOME}/.ssh/id_rsa"
SSH_OPTIONS=(
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=10
)

REMOTE_REPO_DIR="/home/ec2-user/KMeans-StarPU-Heterogeneous-Analysis"

CONTROL_IPS_RAW=""
MPI_IPS_RAW=""
PREFERRED_MASTER_IP="10.0.0.10"
CONTROL_MASTER_IP=""
SLOTS_PER_HOST=1

BUILD_TARGET="starpu"
BUILD_ARGS=""
BUILD_CLEAN_FIRST=0

RUN_TARGET="starpu"
RUN_ARGS="input.txt 50 output_bench/run 32 0 42 30"
RUN_NP=""
RUN_HOSTFILE=""
MPIRUN_BIN="mpirun"
MPIRUN_ARGS=""
MPIRUN_RSH_ARGS="-i ~/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=accept-new"
AUTO_PREPARE_MPI_SSH=1
AUTO_RESOLVE_MPI_PRIVATE_IPS="${AUTO_RESOLVE_MPI_PRIVATE_IPS:-1}"
REMOTE_TRACE_DIR=""
TRACE_TOOL_BIN="starpu_fxt_tool"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--env FILE] [--mpirun-args "args"] <command> [args]

Commands:
  set-ips <control-ips-raw> [mpi-ips-raw]
      Parse and save IPs from raw text.
      Supported examples:
      ['123.123.123.123','111.111.111.111']
      123.123.123.123,111.111.111.111
      If mpi-ips-raw is omitted, private MPI IPs are auto-resolved over SSH.

  show
      Show saved/control/mpi IPs and selected launch/rank0 hosts.

  sync
      Rsync local repository to all control IPs.

  build [openmp|starpu]
      Build target on all control IPs. Default from BUILD_TARGET.
      Flags:
      --clean             run make clean before build on each node
      --no-clean          force no clean for this invocation
      Optional trailing args override BUILD_ARGS for this invocation.

  hostfile
      Generate hostfile at: ${HOSTFILE_PATH}

  fetch-traces [session-name]
      Fetch StarPU FxT trace files from all control hosts into:
      ${LOCAL_TRACES_DIR}/<session-name>
      Options:
      --remote-dir <path>  override remote trace directory on nodes

  condense-traces [session-name|session-path]
      Run ${TRACE_TOOL_BIN} over fetched raw trace files in session and create:
      <session>/condensed

  starvz [session-name|session-path]
      Run starvz on a trace session. Default: latest under ${LOCAL_TRACES_DIR}

  open-starvz-pdf [session-name|session-path]
      Open newest PDF in a trace session. Default: latest under ${LOCAL_TRACES_DIR}

  exec-launch <command...>
      Execute a custom shell command on the launch host inside REMOTE_REPO_DIR.

  exec-all <command...>
      Execute a custom shell command on all control hosts inside REMOTE_REPO_DIR.

  run [openmp|starpu]
      Run mpirun on one launch host using generated hostfile.
      Supports:
      --mpirun-args "..."  (override MPIRUN_ARGS for this run)
      --run-args "..."     (override RUN_ARGS for this run)
      --no-prepare-ssh     skip automatic MPI SSH prep for this run
      --                   (remaining text treated as nbody run args)

  all [openmp|starpu]
      sync + build + run

Examples:
  ./cluster-config/cluster-mpi.sh set-ips "['123.123.123.123','111.111.111.111']"
  ./cluster-config/cluster-mpi.sh set-ips "123.123.123.123,111.111.111.111"
  ./cluster-config/cluster-mpi.sh set-ips "123.123.123.123,111.111.111.111" "10.10.0.10,10.10.0.11"
  ./cluster-config/cluster-mpi.sh sync
  ./cluster-config/cluster-mpi.sh build starpu "USE_MPI=1 USE_CUDA=0 DEBUG=1"
  ./cluster-config/cluster-mpi.sh build starpu --clean "USE_MPI=1 USE_CUDA=0 DEBUG=1"
  ./cluster-config/cluster-mpi.sh fetch-traces
  ./cluster-config/cluster-mpi.sh condense-traces
  ./cluster-config/cluster-mpi.sh fetch-traces 04032026-203208
  ./cluster-config/cluster-mpi.sh condense-traces 04032026-203208
  ./cluster-config/cluster-mpi.sh starvz
  ./cluster-config/cluster-mpi.sh starvz 04032026-203208
  ./cluster-config/cluster-mpi.sh open-starvz-pdf
  ./cluster-config/cluster-mpi.sh open-starvz-pdf 04032026-203208
  ./cluster-config/cluster-mpi.sh exec-launch "make compare"
  ./cluster-config/cluster-mpi.sh exec-all "rm -rf N-Body-Problem-StarPU-OpenMP/build-*"
  ./cluster-config/cluster-mpi.sh run starpu "--mpi --cpu --exp 18"
  ./cluster-config/cluster-mpi.sh run starpu --mpirun-args "--bind-to board" --run-args "--mpi --cpu --exp 18"
  ./cluster-config/cluster-mpi.sh --mpirun-args "--bind-to board" all starpu
USAGE
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

require_cmd() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "Missing required command: $c" >&2
    exit 1
  fi
}

load_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  fi
}

parse_global_flags() {
  while (($# > 0)); do
    case "$1" in
      --env)
        ENV_FILE="$2"
        shift 2
        ;;
      --mpirun-args)
        MPIRUN_ARGS="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        break
        ;;
    esac
  done
  GLOBAL_ARGC=$#
  GLOBAL_ARGV=("$@")
}

extract_ips() {
  local raw="$1"
  raw="$(trim "$raw")"
  [[ -z "$raw" ]] && return 0

  local payload="$raw"
  if [[ "$raw" =~ ^\[.*\]$ ]]; then
    payload="${raw:1:${#raw}-2}"
  elif [[ ! "$raw" =~ ^[0-9.,[:space:]\'\"]+$ ]]; then
    echo "Unsupported IP format. Use \"['ip1','ip2']\" or \"ip1,ip2\"." >&2
    return 1
  fi

  if ! printf '%s\n' "$payload" | grep -Eq "^[0-9.,[:space:]'\"]+$"; then
    echo "Unsupported IP format. Use \"['ip1','ip2']\" or \"ip1,ip2\"." >&2
    return 1
  fi

  payload="$(printf '%s' "$payload" | tr -d '[:space:]')"
  [[ -z "$payload" ]] && return 0

  local -a tokens=()
  IFS=',' read -r -a tokens <<<"$payload"

  local token ip
  local o1 o2 o3 o4
  local printed=","
  for token in "${tokens[@]}"; do
    ip="$token"
    ip="${ip%\'}"
    ip="${ip#\'}"
    ip="${ip%\"}"
    ip="${ip#\"}"

    [[ -z "$ip" ]] && continue
    if [[ ! "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
      echo "Invalid IP token: ${ip}" >&2
      return 1
    fi

    IFS='.' read -r o1 o2 o3 o4 <<<"$ip"
    if ((o1 > 255 || o2 > 255 || o3 > 255 || o4 > 255)); then
      echo "Invalid IP token: ${ip}" >&2
      return 1
    fi

    case "$printed" in
      *,"$ip",*) ;;
      *)
        printf '%s\n' "$ip"
        printed="${printed}${ip},"
        ;;
    esac
  done
}

save_ips_to_file() {
  local path="$1"
  shift
  mkdir -p "$STATE_DIR"
  : > "$path"
  local ip
  for ip in "$@"; do
    printf '%s\n' "$ip" >> "$path"
  done
}

load_ips_from_file() {
  local path="$1"
  local -n out_arr_ref="$2"
  out_arr_ref=()
  if [[ -f "$path" ]]; then
    mapfile -t out_arr_ref < "$path"
  fi
}

contains_ip() {
  local needle="$1"
  shift
  local ip
  for ip in "$@"; do
    if [[ "$ip" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_ips() {
  CONTROL_IPS=()
  MPI_IPS=()
  MPI_IP_SOURCE=""

  if [[ -n "${CONTROL_IPS_RAW:-}" ]]; then
    local control_parsed
    if ! control_parsed="$(extract_ips "$CONTROL_IPS_RAW")"; then
      return 1
    fi
    mapfile -t CONTROL_IPS < <(printf '%s\n' "$control_parsed" | sed '/^$/d')
  else
    load_ips_from_file "$CONTROL_IPS_FILE" CONTROL_IPS
  fi

  if [[ -n "${MPI_IPS_RAW:-}" ]]; then
    local mpi_parsed
    if ! mpi_parsed="$(extract_ips "$MPI_IPS_RAW")"; then
      return 1
    fi
    mapfile -t MPI_IPS < <(printf '%s\n' "$mpi_parsed" | sed '/^$/d')
    MPI_IP_SOURCE="env-mpi-raw"
  else
    load_ips_from_file "$MPI_IPS_FILE" MPI_IPS
    if ((${#MPI_IPS[@]} > 0)); then
      MPI_IP_SOURCE="saved-mpi-file"
    fi
  fi

  if ((${#CONTROL_IPS[@]} == 0)); then
    echo "No control IPs found. Run: $0 set-ips \"['ip1','ip2']\" or \"$0 set-ips ip1,ip2\"" >&2
    return 1
  fi

  if ((${#MPI_IPS[@]} == 0)); then
    if [[ "${AUTO_RESOLVE_MPI_PRIVATE_IPS}" -eq 1 ]]; then
      local resolved_private
      if ! resolved_private="$(resolve_private_ips_from_control)"; then
        return 1
      fi
      mapfile -t MPI_IPS < <(printf '%s\n' "$resolved_private" | sed '/^$/d')
      if ((${#MPI_IPS[@]} == 0)); then
        echo "Failed to auto-resolve private MPI IPs." >&2
        return 1
      fi
      MPI_IP_SOURCE="auto-resolved-private"
    else
      MPI_IPS=("${CONTROL_IPS[@]}")
      MPI_IP_SOURCE="fallback-control-public"
    fi
  fi
}

pick_launch_host() {
  if [[ -n "${CONTROL_MASTER_IP:-}" ]]; then
    if contains_ip "$CONTROL_MASTER_IP" "${CONTROL_IPS[@]}"; then
      printf '%s' "$CONTROL_MASTER_IP"
      return
    fi
    echo "CONTROL_MASTER_IP=${CONTROL_MASTER_IP} is not in control IP list." >&2
    exit 1
  fi

  if [[ -n "${PREFERRED_MASTER_IP:-}" ]] && contains_ip "$PREFERRED_MASTER_IP" "${CONTROL_IPS[@]}"; then
    printf '%s' "$PREFERRED_MASTER_IP"
    return
  fi

  printf '%s' "${CONTROL_IPS[0]}"
}

ordered_mpi_ips_for_rank0() {
  if [[ -n "${PREFERRED_MASTER_IP:-}" ]] && contains_ip "$PREFERRED_MASTER_IP" "${MPI_IPS[@]}"; then
    printf '%s\n' "$PREFERRED_MASTER_IP"
    local ip
    for ip in "${MPI_IPS[@]}"; do
      [[ "$ip" == "$PREFERRED_MASTER_IP" ]] && continue
      printf '%s\n' "$ip"
    done
    return
  fi

  printf '%s\n' "${MPI_IPS[@]}"
}

resolve_private_ips_from_control() {
  ensure_ssh_key_present
  build_ssh_base

  local pub_ip
  local private_ip
  local o1 o2 o3 o4
  for pub_ip in "${CONTROL_IPS[@]}"; do
    if ! private_ip="$(
      ssh_exec "$pub_ip" \
        "ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if (\$i==\"src\") {print \$(i+1); exit}}'"
    )"; then
      echo "SSH-based private IP resolution failed for ${pub_ip}." >&2
      return 1
    fi

    private_ip="$(printf '%s\n' "$private_ip" | tr -d '\r' | sed '/^$/d' | head -n1)"
    if [[ ! "$private_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
      echo "Could not resolve a valid private IP for ${pub_ip}. Got: ${private_ip}" >&2
      return 1
    fi

    IFS='.' read -r o1 o2 o3 o4 <<<"$private_ip"
    if ((o1 > 255 || o2 > 255 || o3 > 255 || o4 > 255)); then
      echo "Resolved invalid private IP for ${pub_ip}: ${private_ip}" >&2
      return 1
    fi

    printf '%s\n' "$private_ip"
  done
}

resolve_latest_trace_session_dir() {
  local latest
  latest="$(find "${LOCAL_TRACES_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

resolve_trace_session_dir() {
  local arg="${1:-}"
  local dir=""

  if [[ -z "$arg" ]]; then
    if ! dir="$(resolve_latest_trace_session_dir)"; then
      echo "No trace sessions found under ${LOCAL_TRACES_DIR}" >&2
      return 1
    fi
  elif [[ -d "$arg" ]]; then
    dir="$arg"
  elif [[ -d "${LOCAL_TRACES_DIR}/${arg}" ]]; then
    dir="${LOCAL_TRACES_DIR}/${arg}"
  else
    echo "Trace session not found: ${arg}" >&2
    return 1
  fi

  printf '%s\n' "$dir"
}

resolve_starvz_tools_dir() {
  require_cmd Rscript

  local tools
  tools="$(Rscript -e 'cat(system.file("tools/", package = "starvz"), sep="\n")')"
  tools="$(printf '%s\n' "$tools" | sed '/^$/d' | head -n1)"
  if [[ -z "$tools" ]]; then
    echo "Could not locate starvz tools. Is the 'starvz' R package installed?" >&2
    return 1
  fi
  if [[ ! -x "${tools}/starvz" ]]; then
    echo "starvz executable not found at: ${tools}/starvz" >&2
    return 1
  fi

  printf '%s\n' "$tools"
}

find_latest_pdf_in_dir() {
  local dir="$1"
  find "$dir" -type f -iname '*.pdf' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n1 \
    | cut -d' ' -f2-
}

detect_remote_trace_dir() {
  local ip="$1"
  local override="${2:-}"

  if [[ -n "$override" ]]; then
    if ssh_exec "$ip" "[ -d $(printf '%q' "$override") ]"; then
      printf '%s\n' "$override"
      return 0
    fi
    echo "Remote trace dir does not exist on ${ip}: ${override}" >&2
    return 1
  fi

  ssh_exec "$ip" '
    for d in "${STARPU_FXT_PREFIX:-}" "$HOME/starpu_traces/fxt" "$HOME/starpu-traces/fxt"; do
      [ -n "$d" ] || continue
      if [ -d "$d" ]; then
        printf "%s\n" "$d"
        exit 0
      fi
    done
    exit 0
  ' | sed '/^$/d' | head -n1
}

default_np() {
  local n_hosts=${#MPI_IPS[@]}
  printf '%d' "$((n_hosts * SLOTS_PER_HOST))"
}

build_ssh_base() {
  SSH_BASE=(ssh -p "$SSH_PORT")
  SCP_BASE=(scp -P "$SSH_PORT")

  if [[ -n "${SSH_KEY:-}" ]]; then
    SSH_BASE+=( -i "$SSH_KEY" )
    SCP_BASE+=( -i "$SSH_KEY" )
  fi

  if ((${#SSH_OPTIONS[@]} > 0)); then
    SSH_BASE+=( "${SSH_OPTIONS[@]}" )
    SCP_BASE+=( "${SSH_OPTIONS[@]}" )
  fi
}

ssh_exec() {
  local ip="$1"
  shift
  local remote_cmd="$*"
  "${SSH_BASE[@]}" "${SSH_USER}@${ip}" "source /etc/profile >/dev/null 2>&1 || true; source /etc/profile.d/openmpi-path.sh >/dev/null 2>&1 || true; source ~/.bashrc >/dev/null 2>&1 || true; ${remote_cmd}"
}

ensure_line_in_remote_file() {
  local ip="$1"
  local line="$2"
  local file="$3"
  local line_b64
  line_b64="$(printf '%s' "$line" | base64 -w0)"
  ssh_exec "$ip" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ${file} && chmod 600 ${file} && line=\$(printf '%s' '${line_b64}' | base64 -d) && grep -qxF \"\$line\" ${file} || printf '%s\n' \"\$line\" >> ${file}"
}

prepare_mpi_ssh() {
  local launch_host="$1"

  echo "[run] preparing MPI SSH from launch host ${launch_host} ..."

  ssh_exec "$launch_host" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && [ -f ~/.ssh/id_ed25519_mpi ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519_mpi"
  local launch_pub
  launch_pub="$(ssh_exec "$launch_host" "cat ~/.ssh/id_ed25519_mpi.pub")"
  if [[ -z "$launch_pub" ]]; then
    echo "Failed to read launch host MPI SSH public key." >&2
    exit 1
  fi

  local ip
  local idx
  local provision_ip
  for idx in "${!MPI_IPS[@]}"; do
    ip="${MPI_IPS[$idx]}"
    provision_ip="$ip"
    # If control and mpi lists are aligned, use control/public IP for provisioning.
    # This allows private MPI IPs without requiring private reachability from local machine.
    if ((${#CONTROL_IPS[@]} == ${#MPI_IPS[@]})); then
      provision_ip="${CONTROL_IPS[$idx]}"
    fi
    ensure_line_in_remote_file "$provision_ip" "$launch_pub" "~/.ssh/authorized_keys"
  done

  # Pre-seed launch host known_hosts to avoid host verification prompts.
  ssh_exec "$launch_host" "touch ~/.ssh/known_hosts && chmod 600 ~/.ssh/known_hosts"
  for ip in "${MPI_IPS[@]}"; do
    ssh_exec "$launch_host" "ssh-keygen -R ${ip} >/dev/null 2>&1 || true; ssh-keyscan -H ${ip} >> ~/.ssh/known_hosts 2>/dev/null || true"
  done
}

ensure_ssh_key_present() {
  if [[ -n "${SSH_KEY:-}" && ! -f "${SSH_KEY}" ]]; then
    echo "SSH key file not found: ${SSH_KEY}" >&2
    exit 1
  fi
}

cmd_set_ips() {
  if (($# < 1)); then
    echo "set-ips requires at least one raw IP text argument." >&2
    exit 1
  fi

  local control_raw="$1"
  local mpi_raw="${2:-}"

  local -a control_ips=()
  local -a mpi_ips=()

  local control_parsed mpi_parsed
  if ! control_parsed="$(extract_ips "$control_raw")"; then
    exit 1
  fi
  if [[ -n "$mpi_raw" ]]; then
    if ! mpi_parsed="$(extract_ips "$mpi_raw")"; then
      exit 1
    fi
  fi
  mapfile -t control_ips < <(printf '%s\n' "$control_parsed" | sed '/^$/d')
  if [[ -n "$mpi_raw" ]]; then
    mapfile -t mpi_ips < <(printf '%s\n' "$mpi_parsed" | sed '/^$/d')
  fi

  if ((${#control_ips[@]} == 0)); then
    echo "Could not parse any IPs from control raw text." >&2
    exit 1
  fi

  save_ips_to_file "$CONTROL_IPS_FILE" "${control_ips[@]}"
  if [[ -n "$mpi_raw" ]]; then
    save_ips_to_file "$MPI_IPS_FILE" "${mpi_ips[@]}"
    echo "Saved MPI IPs to ${MPI_IPS_FILE}"
  else
    : > "$MPI_IPS_FILE"
    if [[ "${AUTO_RESOLVE_MPI_PRIVATE_IPS}" -eq 1 ]]; then
      echo "No MPI IPs provided; will auto-resolve private MPI IPs from control public IPs."
    else
      echo "No MPI IPs provided; MPI IPs will default to control IPs."
    fi
  fi

  echo "Saved control IPs to ${CONTROL_IPS_FILE}"
  if ! cmd_show; then
    echo "Saved IPs, but unable to resolve/show MPI IPs now. Check SSH connectivity to control nodes for auto-resolution." >&2
  fi
}

cmd_show() {
  if ! resolve_ips; then
    return 1
  fi

  local launch
  launch="$(pick_launch_host)"

  local rank0
  rank0="$(ordered_mpi_ips_for_rank0 | head -n1)"

  echo "Control IPs (${#CONTROL_IPS[@]}):"
  printf '  - %s\n' "${CONTROL_IPS[@]}"

  echo "MPI IPs (${#MPI_IPS[@]}):"
  printf '  - %s\n' "${MPI_IPS[@]}"
  if [[ -n "${MPI_IP_SOURCE:-}" ]]; then
    echo "MPI IP source: ${MPI_IP_SOURCE}"
  fi

  echo "Launch host (where mpirun runs): ${launch}"
  echo "Rank0 preferred hostfile entry: ${rank0}"
}

cmd_sync() {
  resolve_ips
  require_cmd rsync
  ensure_ssh_key_present
  build_ssh_base

  local rsync_ssh=(ssh -p "$SSH_PORT")
  if [[ -n "${SSH_KEY:-}" ]]; then
    rsync_ssh+=( -i "$SSH_KEY" )
  fi
  if ((${#SSH_OPTIONS[@]} > 0)); then
    rsync_ssh+=( "${SSH_OPTIONS[@]}" )
  fi

  local rsync_ssh_cmd
  rsync_ssh_cmd="${rsync_ssh[*]}"

  local ip
  for ip in "${CONTROL_IPS[@]}"; do
    echo "[sync] ${ip}"
    ssh_exec "$ip" "mkdir -p $(printf '%q' "$REMOTE_REPO_DIR")"
    rsync -az \
      --exclude '.git/' \
      --exclude 'cluster-config/.cluster/' \
      --exclude 'results/' \
      --exclude 'input.txt' \
      --exclude 'input_sanity_10k.txt' \
      --exclude 'logs/' \
      --exclude 'output_bench/' \
      --exclude 'output_sanity/' \
      -e "$rsync_ssh_cmd" \
      "${REPO_ROOT}/" "${SSH_USER}@${ip}:${REMOTE_REPO_DIR}/"
  done
}

cmd_build() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  local target="${1:-$BUILD_TARGET}"
  shift || true
  target="$(trim "$target")"
  if [[ "$target" != "openmp" && "$target" != "starpu" ]]; then
    echo "build target must be openmp or starpu" >&2
    exit 1
  fi

  local do_clean="$BUILD_CLEAN_FIRST"
  while (($# > 0)); do
    case "$1" in
      --clean)
        do_clean=1
        shift
        ;;
      --no-clean)
        do_clean=0
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

  local build_args="$BUILD_ARGS"
  if (($# > 0)); then
    build_args="$*"
  fi

  local build_args_b64
  build_args_b64="$(printf '%s' "$build_args" | base64 -w0)"

  local ip
  for ip in "${CONTROL_IPS[@]}"; do
    echo "[build:$target] ${ip}"
    ssh_exec "$ip" "\
      cd $(printf '%q' "$REMOTE_REPO_DIR") && \
      build_args=\$(printf '%s' '${build_args_b64}' | base64 -d) && \
      state_file=.cluster-mpi-last-build-${target}.args && \
      need_clean=${do_clean} && \
      prev_args='' && \
      if [ -f \"\$state_file\" ]; then prev_args=\$(cat \"\$state_file\"); fi && \
      if [ \"\$prev_args\" != \"\$build_args\" ]; then \
        echo \"[build:${target}] build args changed; running ${target}-clean\"; \
        need_clean=1; \
      fi && \
      if [ \"\$need_clean\" -eq 1 ]; then \
        if [ \"${target}\" = \"openmp\" ]; then cd src/openmp && make clean && cd - ; else make clean; fi; \
      fi && \
      if [ \"${target}\" = \"openmp\" ]; then \
        cd src/openmp && make GPU=1 && cd - ; \
      else \
        make \$build_args; \
      fi && \
      printf '%s' \"\$build_args\" > \"\$state_file\" \
    "
  done
}

cmd_disable_ht() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  local ip
  for ip in "${CONTROL_IPS[@]}"; do
    echo "[disable-ht] ${ip}"
    ssh_exec "$ip" "echo off | sudo tee /sys/devices/system/cpu/smt/control && echo 'HT disabled. CPUs: '$(nproc)"
  done
}

cmd_generate_input() {
  local n="${1:-300000000}"
  local d="${2:-2}"
  local seed="${3:-42}"
  local filename="${4:-input.txt}"
  local procs="${5:-8}"

  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  local launch_host
  launch_host="$(pick_launch_host)"

  echo "[generate-input] Gerando ${n} pontos no nó mestre ${launch_host}..."
  ssh_exec "$launch_host" \
    "cd $(printf '%q' "$REMOTE_REPO_DIR") && python3 generate_input.py ${n} ${d} ${seed} ${filename} ${procs}"
  echo "[generate-input] Concluido: ${filename}"
}

cmd_setup() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  local ip
  for ip in "${CONTROL_IPS[@]}"; do
    echo "[setup] Installing NVIDIA HPC SDK on ${ip}..."
    ssh_exec "$ip" "
      if command -v nvc++ >/dev/null 2>&1; then
        echo '[setup] nvc++ already installed, skipping.'
      else
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/hpc-sdk/rhel/nvhpc.repo &&
        sudo dnf install -y nvhpc-26-3 &&
        echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:\$PATH' >> ~/.bashrc &&
        source ~/.bashrc &&
        echo '[setup] nvc++ installed successfully.'
      fi
    "
  done
  echo "[setup] Done on all nodes."
}

cmd_hostfile() {
  resolve_ips
  mkdir -p "$STATE_DIR"

  : > "$HOSTFILE_PATH"
  local ip
  while IFS= read -r ip; do
    [[ -z "$ip" ]] && continue
    printf '%s slots=%s\n' "$ip" "$SLOTS_PER_HOST" >> "$HOSTFILE_PATH"
  done < <(ordered_mpi_ips_for_rank0)

  echo "Generated hostfile: ${HOSTFILE_PATH}"
  cat "$HOSTFILE_PATH"
}

cmd_fetch_traces() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  local session_name=""
  local remote_dir_override="${REMOTE_TRACE_DIR:-}"
  while (($# > 0)); do
    case "$1" in
      --remote-dir)
        remote_dir_override="$2"
        shift 2
        ;;
      --)
        shift
        if (($# > 0)); then
          session_name="$1"
          shift
        fi
        ;;
      *)
        if [[ -z "$session_name" ]]; then
          session_name="$1"
          shift
        else
          echo "Unexpected argument for fetch-traces: $1" >&2
          exit 1
        fi
        ;;
    esac
  done

  if [[ -z "$session_name" ]]; then
    session_name="$(date +%d%m%Y-%H%M%S)"
  fi

  local session_dir="${LOCAL_TRACES_DIR}/${session_name}"
  mkdir -p "$session_dir"

  local manifest="${session_dir}/manifest.txt"
  {
    echo "session=${session_name}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "trace_source=remote_starpu_fxt_prefix_or_home_starpu_traces_fxt"
  } > "$manifest"

  local ip remote_dir ip_tag
  local idx=1
  local fetched_nodes=0
  for ip in "${CONTROL_IPS[@]}"; do
    ip_tag="${ip//./-}"

    remote_dir="$(detect_remote_trace_dir "$ip" "$remote_dir_override" || true)"
    if [[ -z "$remote_dir" ]]; then
      echo "[fetch-traces] WARN: no remote trace directory found on ${ip}; skipping."
      printf 'node=%s control_ip=%s remote_dir=<not-found> files=0\n' "$idx" "$ip" >> "$manifest"
      idx=$((idx + 1))
      continue
    fi

    echo "[fetch-traces] ${ip} (${remote_dir}) -> ${session_dir}"
    local -a remote_files=()
    mapfile -t remote_files < <(
      ssh_exec "$ip" "find $(printf '%q' "$remote_dir") -maxdepth 1 -type f \\( -name 'prof_file_*' -o -name 'proc_file_*' \\) -printf '%f\n' | sort"
    )

    if ((${#remote_files[@]} == 0)); then
      echo "[fetch-traces] WARN: no prof_file_*/proc_file_* found on ${ip}; skipping."
      printf 'node=%s control_ip=%s remote_dir=%s files=0\n' "$idx" "$ip" "$remote_dir" >> "$manifest"
      idx=$((idx + 1))
      continue
    fi

    local remote_base remote_path local_base local_path
    local copied=0
    for remote_base in "${remote_files[@]}"; do
      remote_path="${remote_dir%/}/${remote_base}"
      case "$remote_base" in
        prof_file_*)
          local_base="prof_file_${ip_tag}-${remote_base#prof_file_}"
          ;;
        proc_file_*)
          local_base="proc_file_${ip_tag}-${remote_base#proc_file_}"
          ;;
        *)
          local_base="${ip_tag}-${remote_base}"
          ;;
      esac
      local_path="${session_dir}/${local_base}"
      if "${SCP_BASE[@]}" "${SSH_USER}@${ip}:${remote_path}" "${local_path}"; then
        copied=$((copied + 1))
      else
        echo "[fetch-traces] WARN: failed to fetch ${remote_path}"
      fi
    done

    if [[ "$copied" -gt 0 ]]; then
      printf 'node=%s control_ip=%s remote_dir=%s files=%s\n' "$idx" "$ip" "$remote_dir" "$copied" >> "$manifest"
      fetched_nodes=$((fetched_nodes + 1))
    else
      printf 'node=%s control_ip=%s remote_dir=%s files=0 scp=failed\n' "$idx" "$ip" "$remote_dir" >> "$manifest"
    fi
    idx=$((idx + 1))
  done

  if [[ "$fetched_nodes" -eq 0 ]]; then
    echo "No traces fetched. Check remote trace directory and SSH connectivity." >&2
    exit 1
  fi

  echo "Fetched traces into: ${session_dir}"
  echo "Next: ./cluster-config/cluster-mpi.sh condense-traces ${session_name}"
}

cmd_condense_traces() {
  local trace_arg="${1:-}"
  if (($# > 1)); then
    echo "condense-traces accepts at most one argument: [session-name|session-path]" >&2
    exit 1
  fi

  require_cmd "${TRACE_TOOL_BIN}"

  local session_dir
  if ! session_dir="$(resolve_trace_session_dir "$trace_arg")"; then
    exit 1
  fi

  mapfile -t trace_inputs < <(
    find "${session_dir}" -maxdepth 1 -type f \( -name 'prof_file_*' -o -name 'proc_file_*' \) | sort
  )
  if ((${#trace_inputs[@]} == 0)); then
    echo "No prof_file_*/proc_file_* files found in: ${session_dir}" >&2
    exit 1
  fi

  echo "[condense-traces] session: ${session_dir}"
  echo "[condense-traces] input files: ${#trace_inputs[@]}"
  (
    cd "$session_dir"
    "${TRACE_TOOL_BIN}" -i "${trace_inputs[@]}"
  )

  local condensed_file="${session_dir}/condensed"
  {
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool=${TRACE_TOOL_BIN}"
    echo "inputs=${#trace_inputs[@]}"
  } > "$condensed_file"

  echo "Condensed trace marker created: ${condensed_file}"
}

cmd_starvz() {
  local trace_arg="${1:-}"
  if (($# > 1)); then
    echo "starvz accepts at most one argument: [session-name|session-path]" >&2
    exit 1
  fi

  local session_dir
  if ! session_dir="$(resolve_trace_session_dir "$trace_arg")"; then
    exit 1
  fi

  local tools
  if ! tools="$(resolve_starvz_tools_dir)"; then
    exit 1
  fi

  echo "[starvz] tools: ${tools}"
  echo "[starvz] trace: ${session_dir}"
  "${tools}/starvz" "$session_dir"
}

cmd_open_starvz_pdf() {
  local trace_arg="${1:-}"
  if (($# > 1)); then
    echo "open-starvz-pdf accepts at most one argument: [session-name|session-path]" >&2
    exit 1
  fi

  local session_dir
  if ! session_dir="$(resolve_trace_session_dir "$trace_arg")"; then
    exit 1
  fi

  local pdf_path
  pdf_path="$(find_latest_pdf_in_dir "$session_dir")"
  if [[ -z "$pdf_path" ]]; then
    echo "No PDF found in: ${session_dir}" >&2
    echo "Run: ./cluster-config/cluster-mpi.sh starvz ${trace_arg:-}" >&2
    exit 1
  fi

  echo "[open-starvz-pdf] ${pdf_path}"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$pdf_path" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then
    open "$pdf_path" >/dev/null 2>&1 &
  elif command -v gio >/dev/null 2>&1; then
    gio open "$pdf_path" >/dev/null 2>&1 &
  else
    echo "No opener found (xdg-open/open/gio). File path: ${pdf_path}" >&2
    exit 1
  fi
}

cmd_exec_launch() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  if (($# == 0)); then
    echo "exec-launch requires a command string." >&2
    exit 1
  fi

  local launch_host
  launch_host="$(pick_launch_host)"
  local custom_cmd="$*"

  echo "[exec-launch] ${launch_host}: ${custom_cmd}"
  ssh_exec "$launch_host" "cd $(printf '%q' "$REMOTE_REPO_DIR") && ${custom_cmd}"
}

cmd_exec_all() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  if (($# == 0)); then
    echo "exec-all requires a command string." >&2
    exit 1
  fi

  local custom_cmd="$*"
  local ip
  for ip in "${CONTROL_IPS[@]}"; do
    echo "[exec-all] ${ip}: ${custom_cmd}"
    ssh_exec "$ip" "cd $(printf '%q' "$REMOTE_REPO_DIR") && ${custom_cmd}"
  done
}

cmd_run() {
  resolve_ips
  ensure_ssh_key_present
  build_ssh_base

  local target="${1:-$RUN_TARGET}"
  shift || true
  target="$(trim "$target")"
  if [[ "$target" != "openmp" && "$target" != "starpu" ]]; then
    echo "run target must be openmp or starpu" >&2
    exit 1
  fi

  local run_args="$RUN_ARGS"
  local mpirun_args="$MPIRUN_ARGS"
  local do_prepare_ssh="$AUTO_PREPARE_MPI_SSH"
  while (($# > 0)); do
    case "$1" in
      --mpirun-args)
        mpirun_args="$2"
        shift 2
        ;;
      --run-args)
        run_args="$2"
        shift 2
        ;;
      --no-prepare-ssh)
        do_prepare_ssh=0
        shift
        ;;
      --)
        shift
        if (($# > 0)); then
          run_args="$*"
        fi
        break
        ;;
      *)
        # Backward compatibility: trailing positional text is nbody runtime args.
        run_args="$*"
        break
        ;;
    esac
  done

  local launch_host
  launch_host="$(pick_launch_host)"

  local local_hostfile
  if [[ -n "${RUN_HOSTFILE:-}" ]]; then
    local_hostfile="$RUN_HOSTFILE"
    if [[ ! -f "$local_hostfile" ]]; then
      echo "RUN_HOSTFILE not found: ${local_hostfile}" >&2
      exit 1
    fi
  else
    cmd_hostfile >/dev/null
    local_hostfile="$HOSTFILE_PATH"
  fi

  local np
  if [[ -n "${RUN_NP:-}" ]]; then
    np="$RUN_NP"
  else
    np="$(default_np)"
  fi

  local remote_hostfile="${REMOTE_REPO_DIR}/cluster-config/hostfile.generated"

  echo "[run] launch host: ${launch_host}"
  echo "[run] hostfile: ${local_hostfile}"
  echo "[run] np: ${np}"
  echo "[run] mpirun args: ${mpirun_args:-<none>}"

  if [[ "$do_prepare_ssh" -eq 1 ]]; then
    prepare_mpi_ssh "$launch_host"
  fi

  "${SCP_BASE[@]}" "$local_hostfile" "${SSH_USER}@${launch_host}:${remote_hostfile}"

  ssh_exec "$launch_host" \
    "cd $(printf '%q' "$REMOTE_REPO_DIR") && \
      if [ \"${target}\" = \"openmp\" ]; then \
        binary='./src/openmp/kmeans_openmp'; \
      else \
        export STARPU_SCHED=dmda STARPU_NOPENCL=1 STARPU_RESERVE_NCPU=1; \
        binary='./kmeans_starpu'; \
      fi && \
      ${MPIRUN_BIN} --mca plm_rsh_args \"${MPIRUN_RSH_ARGS}\" ${mpirun_args} --hostfile $(printf '%q' "$remote_hostfile") -np ${np} \$binary ${run_args}"
}

cmd_all() {
  local target="${1:-$RUN_TARGET}"
  cmd_sync
  cmd_build "$target"
  cmd_run "$target"
}

main() {
  parse_global_flags "$@"
  set -- "${GLOBAL_ARGV[@]}"

  load_env_file

  if (($# == 0)); then
    usage
    exit 1
  fi

  local cmd="$1"
  shift

  case "$cmd" in
    set-ips) cmd_set_ips "$@" ;;
    show) cmd_show ;;
    setup) cmd_setup ;;
    disable-ht) cmd_disable_ht ;;
    generate-input) cmd_generate_input "$@" ;;
    sync) cmd_sync ;;
    build) cmd_build "$@" ;;
    hostfile) cmd_hostfile ;;
    fetch-traces) cmd_fetch_traces "$@" ;;
    condense-traces) cmd_condense_traces "$@" ;;
    starvz) cmd_starvz "$@" ;;
    open-starvz-pdf) cmd_open_starvz_pdf "$@" ;;
    exec-launch) cmd_exec_launch "$@" ;;
    exec-all) cmd_exec_all "$@" ;;
    run) cmd_run "$@" ;;
    all) cmd_all "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
