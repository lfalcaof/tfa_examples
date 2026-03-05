#!/usr/bin/env bash
set -euo pipefail

# ============================================
# run_many_equal_toys_v1.sh
# - Runs d02kspipi_toys_v1.py N times
# - Each run uses a different --seed
# - All outputs go to ONE folder under ../../output/
# - Output name per toy: Toy_seedX_Acc_v1
# ============================================

# ----------------------------
# User-facing inputs
# ----------------------------
N="${1:-5}"                              # number of toys/runs (seeds 1..N)
CAMPAIGN="${CAMPAIGN:-Equal_${N}toys_v1}" # output folder name under OUTPUT_BASE_DIR_REL

# ----------------------------
# Configuration
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_REL="${SCRIPT_REL:-d02kspipi_toys_v1.py}"

NTOYS="${NTOYS:-2000000}"
X="${X:-0.004}"
Y="${Y:-0.0064}"
QOP="${QOP:-1.0}"
QOP_PHASE="${QOP_PHASE:-0.0}"

# v1 acceptance knobs (as per your command)
USE_ACCEPTANCE="${USE_ACCEPTANCE:-1}"   # 1=yes, 0=no
RHO="${RHO:-0.8}"
ALPHA="${ALPHA:-1.4}"
EPS_MIN="${EPS_MIN:-0.05}"

# Base output directory (relative to this script location)
OUTPUT_BASE_DIR_REL="${OUTPUT_BASE_DIR_REL:-../../output}"

# ----------------------------
# Anchor to script dir
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Resolve absolute path for output base
OUTPUT_BASE_DIR_ABS="$("${PYTHON_BIN}" - <<PY 2>/dev/null || echo "${OUTPUT_BASE_DIR_REL}"
import os
print(os.path.abspath("${OUTPUT_BASE_DIR_REL}"))
PY
)"

# Campaign directory (single folder for all toys)
CAMPAIGN_DIR="${OUTPUT_BASE_DIR_ABS}/${CAMPAIGN}"
mkdir -p "${CAMPAIGN_DIR}"

# Save a campaign-level config snapshot
cat > "${CAMPAIGN_DIR}/campaign_info.txt" <<EOF
campaign: ${CAMPAIGN}
script_dir: ${SCRIPT_DIR}
python_bin: ${PYTHON_BIN}
script_rel: ${SCRIPT_REL}
N_runs: ${N}
ntoys_per_run: ${NTOYS}
x: ${X}
y: ${Y}
qop: ${QOP}
qop_phase: ${QOP_PHASE}
use_acceptance: ${USE_ACCEPTANCE}
rho: ${RHO}
alpha: ${ALPHA}
eps_min: ${EPS_MIN}
EOF

echo "[INFO] Campaign dir: ${CAMPAIGN_DIR}"
echo "[INFO] Running ${N} toys..."

for ((seed=1; seed<=N; seed++)); do
  toy_name="Toy_seed${seed}_Acc_v1"

  # Keep the same output naming logic:
  # each run uses --output <campaign>/<toy_name>
  out_prefix_rel="${CAMPAIGN}/${toy_name}"

  log_file="${CAMPAIGN_DIR}/${toy_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_REL}"
    --output "${out_prefix_rel}"
    --ntoys "${NTOYS}"
    --x "${X}"
    --y "${Y}"
    --qop "${QOP}"
    --qop_phase "${QOP_PHASE}"
    --seed "${seed}"
  )

  if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
    cmd+=( --use_acceptance --rho "${RHO}" --alpha "${ALPHA}" --eps_min "${EPS_MIN}" )
  fi

  echo "[INFO] (${seed}/${N}) ${toy_name}"

  # Save exact command (quoted safely) for reproducibility
  printf '%q ' "${cmd[@]}" > "${CAMPAIGN_DIR}/${toy_name}_command.sh"
  echo >> "${CAMPAIGN_DIR}/${toy_name}_command.sh"
  chmod +x "${CAMPAIGN_DIR}/${toy_name}_command.sh"

  # Run and capture log per seed
  (
    cd "${SCRIPT_DIR}"
    "${CAMPAIGN_DIR}/${toy_name}_command.sh"
  ) &> "${log_file}"
done

echo "[INFO] Done. All logs and commands are in:"
echo "       ${CAMPAIGN_DIR}"
