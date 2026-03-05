#!/usr/bin/env bash
set -euo pipefail

# ============================================
# run_many_equal_toys_v2.sh
# - Runs d02kspipi_toys_v2.py N times
# - Each run uses a different --seed
# - All outputs go to ONE folder under ../../output/
# - Output name per toy: Toy_seedX_v2
# ============================================
# to Run-> nohup ./run_many_equal_toys_v2.sh 1000 > Equal_1000toys.log 2>&1 & disown
# ----------------------------
# User-facing inputs
# ----------------------------
N="${1:-5}"                            # number of toys/runs (seeds 1..N)
USE_ACCEPTANCE="${USE_ACCEPTANCE:-1}"  # 1=yes, 0=no

# Acceptance type (only used if USE_ACCEPTANCE=1)
# Allowed: selTwoTracks | selD2Kshh | selD2Kshh_TwoTracks
ACC_TYPE="${ACC_TYPE:-selD2Kshh_TwoTracks}"

ACC_TAG="NoACC"
if [[ "$USE_ACCEPTANCE" -eq 1 ]]; then
  ACC_TAG="WithACC_${ACC_TYPE}"
fi

echo "[INFO] Toys generated: ${ACC_TAG}"
CAMPAIGN="${CAMPAIGN:-Equal_${N}toys_v2_${ACC_TAG}}"

# ----------------------------
# Configuration
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_REL="${SCRIPT_REL:-d02kspipi_toys_v2.py}"

NEVENTS="${NEVENTS:-2000000}"
X="${X:-0.004}"
Y="${Y:-0.006}"
QOP="${QOP:-1.0}"
QOP_PHASE="${QOP_PHASE:-0.0}"

# IMPORTANT:
# Use only ONE formatting style.
# If your python uses: hist_pattern.format(i)
# then pattern must be "accDP_tbin_{:02d}" (Python format-style), NOT "accDP_tbin_%02d".
ACC_HIST_PATTERN="${ACC_HIST_PATTERN:-accDP_tbin_{:02d}}"

# Base output directory (relative to this script location)
OUTPUT_BASE_DIR_REL="${OUTPUT_BASE_DIR_REL:-../../output}"

# ----------------------------
# Anchor to script dir
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
  case "${ACC_TYPE}" in
    selTwoTracks)
      ACC_ROOT_DP="../../time_dependent_acceptance/version_selTwoTracks/acc_time_dep_selTwoTracks_dalitz.root"
      ;;
    selD2Kshh)
      ACC_ROOT_DP="../../time_dependent_acceptance/version_selD2Kshh/acc_time_dep_selD2Kshh_dalitz.root"
      ;;
    selD2Kshh_TwoTracks)
      ACC_ROOT_DP="../../time_dependent_acceptance/version_selD2Kshh_TwoTracks/acc_time_dep_selD2Kshh_TwoTracks_dalitz.root"
      ;;
    *)
      echo "[ERROR] Unknown ACC_TYPE='${ACC_TYPE}'" >&2
      echo "        Allowed: selTwoTracks | selD2Kshh | selD2Kshh_TwoTracks" >&2
      exit 3
      ;;
  esac
fi

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
nevents_per_run: ${NEVENTS}
x: ${X}
y: ${Y}
qop: ${QOP}
qop_phase: ${QOP_PHASE}
use_acceptance: ${USE_ACCEPTANCE}
acc_type: ${ACC_TYPE}
acc_root_dp: ${ACC_ROOT_DP}
acc_hist_pattern: ${ACC_HIST_PATTERN}
EOF

# If acceptance pattern contains braces, ensure bash didn't mangle it
# (we keep it as a plain string; no eval, no printf-formatting)
if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
  # quick sanity: must not contain unmatched '}' as per your error
  if [[ "${ACC_HIST_PATTERN}" == *"}"* && "${ACC_HIST_PATTERN}" != *"{"* ]]; then
    echo "[ERROR] ACC_HIST_PATTERN seems to contain '}' but no '{': ${ACC_HIST_PATTERN}" >&2
    echo "        Use something like: accDP_tbin_{:02d}" >&2
    exit 2
  fi

  # sanity: ensure ACC_ROOT_DP file exists (relative to SCRIPT_DIR)
  if [[ ! -f "${SCRIPT_DIR}/${ACC_ROOT_DP}" && ! -f "${ACC_ROOT_DP}" ]]; then
    echo "[ERROR] ACC_ROOT_DP not found: ${ACC_ROOT_DP}" >&2
    echo "        Checked: '${SCRIPT_DIR}/${ACC_ROOT_DP}' and '${ACC_ROOT_DP}'" >&2
    exit 4
  fi
fi

echo "[INFO] Campaign dir: ${CAMPAIGN_DIR}"
echo "[INFO] Running ${N} toys..."

for ((seed=1; seed<=N; seed++)); do
  toy_name="Toy_seed${seed}_v2"

  # Output prefix relative to output base used by python script
  out_prefix_rel="${CAMPAIGN}/${toy_name}"

  log_file="${CAMPAIGN_DIR}/${toy_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_REL}"
    --output "${out_prefix_rel}"
    --nevents "${NEVENTS}"
    --x "${X}"
    --y "${Y}"
    --qop "${QOP}"
    --qop_phase "${QOP_PHASE}"
    --seed "${seed}"
  )

  if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
    cmd+=( --use_acceptance --acc_root_dp "${ACC_ROOT_DP}" --acc_hist_pattern "${ACC_HIST_PATTERN}" )
  fi

  echo "[INFO] (${seed}/${N}) ${toy_name}"
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
