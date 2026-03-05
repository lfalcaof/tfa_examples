#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_toys_v2.sh  (single toy, but structured like run_many_equal_toys_v2.sh)
#
# - Runs d02kspipi_toys_v2.py ONCE
# - Uses the same ACC_TYPE -> ACC_ROOT_DP structure as run_many_equal_toys_v2.sh
# - Output goes to ONE campaign folder under ../../output/
# - Output name: Toy_seed<SEED>_v2
#
# ============================================================

# ----------------------------
# User-facing inputs
# ----------------------------

# Human identifier for this run/campaign (keep simple)
TOY_ID="${TOY_ID:-_No_Mixing}"   # e.g. 001, 014, "ACCtestA"

# Acceptance on/off
USE_ACCEPTANCE="${USE_ACCEPTANCE:-0}"  # 1=yes, 0=no

# Allowed: selTwoTracks | selD2Kshh | selD2Kshh_TwoTracks
ACC_TYPE="${ACC_TYPE:-selD2Kshh_TwoTracks}"

ACC_TAG="NoACC"
if [[ "${USE_ACCEPTANCE}" -eq 1 ]]; then
  ACC_TAG="WithACC_${ACC_TYPE}"
fi

# Optional free-text tag (appears in campaign folder name)
RUN_TAG="${RUN_TAG:-}"   # e.g. "sanity", leave empty if not needed

# ----------------------------
# Configuration
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_REL="${SCRIPT_REL:-d02kspipi_toys_v2.py}"

# Toy config (keep variable names consistent with your python)
NEVENTS="${NEVENTS:-2000000}"
X="${X:-0.000}"
Y="${Y:-0.000}"
QOP="${QOP:-1.0}"
QOP_PHASE="${QOP_PHASE:-0.0}"
SEED="${SEED:-1}"

# IMPORTANT: python uses hist_pattern.format(i), so keep Python-format style
ACC_HIST_PATTERN="${ACC_HIST_PATTERN:-accDP_tbin_{:02d}}"

# Base output directory (relative to this script location)
OUTPUT_BASE_DIR_REL="${OUTPUT_BASE_DIR_REL:-../../output}"

# ----------------------------
# Anchor to script dir
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ----------------------------
# Resolve ACC_ROOT_DP from ACC_TYPE (same as run_many_equal_toys_v2.sh)
# ----------------------------
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

# ----------------------------
# Sanity checks (pattern + acc file exists)
# ----------------------------
if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
  if [[ "${ACC_HIST_PATTERN}" == *"}"* && "${ACC_HIST_PATTERN}" != *"{"* ]]; then
    echo "[ERROR] ACC_HIST_PATTERN seems to contain '}' but no '{': ${ACC_HIST_PATTERN}" >&2
    echo "        Use something like: accDP_tbin_{:02d}" >&2
    exit 2
  fi

  if [[ ! -f "${SCRIPT_DIR}/${ACC_ROOT_DP}" && ! -f "${ACC_ROOT_DP}" ]]; then
    echo "[ERROR] ACC_ROOT_DP not found: ${ACC_ROOT_DP}" >&2
    echo "        Checked: '${SCRIPT_DIR}/${ACC_ROOT_DP}' and '${ACC_ROOT_DP}'" >&2
    exit 4
  fi
fi

# ----------------------------
# Resolve absolute output base dir
# ----------------------------
OUTPUT_BASE_DIR_ABS="$("${PYTHON_BIN}" - <<PY 2>/dev/null || echo "${OUTPUT_BASE_DIR_REL}"
import os
print(os.path.abspath("${OUTPUT_BASE_DIR_REL}"))
PY
)"

# ----------------------------
# Campaign folder naming (NO DATE)
# ----------------------------
# Example:
#   Toy_v2__IDD2Kshh__WithACC_selD2Kshh_TwoTracks
#   Toy_v2__IDD2Kshh__NoACC
CAMPAIGN="Toy_v2__ID${TOY_ID}__${ACC_TAG}"
if [[ -n "${RUN_TAG}" ]]; then
  CAMPAIGN="${CAMPAIGN}__${RUN_TAG}"
fi

CAMPAIGN_DIR="${OUTPUT_BASE_DIR_ABS}/${CAMPAIGN}"
mkdir -p "${CAMPAIGN_DIR}"

# ----------------------------
# Toy naming (single run)
# ----------------------------
TOY_NAME="Toy_seed${SEED}_v2"
OUT_PREFIX_REL="${CAMPAIGN}/${TOY_NAME}"

# ----------------------------
# Build command
# ----------------------------
cmd=(
  "${PYTHON_BIN}" "${SCRIPT_REL}"
  --output "${OUT_PREFIX_REL}"
  --nevents "${NEVENTS}"
  --x "${X}"
  --y "${Y}"
  --qop "${QOP}"
  --qop_phase "${QOP_PHASE}"
  --seed "${SEED}"
)

if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
  cmd+=( --use_acceptance --acc_root_dp "${ACC_ROOT_DP}" --acc_hist_pattern "${ACC_HIST_PATTERN}" )
fi

# Save command
printf '%q ' "${cmd[@]}" > "${CAMPAIGN_DIR}/${TOY_NAME}_command.sh"
echo >> "${CAMPAIGN_DIR}/${TOY_NAME}_command.sh"
chmod +x "${CAMPAIGN_DIR}/${TOY_NAME}_command.sh"

# Save run metadata (similar spirit to the old provenance)
cat > "${CAMPAIGN_DIR}/${TOY_NAME}_run_info.txt" <<EOF
campaign: ${CAMPAIGN}
campaign_dir: ${CAMPAIGN_DIR}
toy_name: ${TOY_NAME}
script_dir: ${SCRIPT_DIR}
python_bin: ${PYTHON_BIN}
python_version: $(${PYTHON_BIN} --version 2>&1 || true)
script_rel: ${SCRIPT_REL}

output_prefix_rel: ${OUT_PREFIX_REL}

params:
  nevents: ${NEVENTS}
  x: ${X}
  y: ${Y}
  qop: ${QOP}
  qop_phase: ${QOP_PHASE}
  seed: ${SEED}

acceptance:
  use_acceptance: ${USE_ACCEPTANCE}
  acc_type: ${ACC_TYPE}
  acc_root_dp: ${ACC_ROOT_DP:-}
  acc_hist_pattern: ${ACC_HIST_PATTERN}
EOF

# JSON manifest (programmatic tracking)
cat > "${CAMPAIGN_DIR}/${TOY_NAME}_manifest.json" <<EOF
{
  "campaign": "${CAMPAIGN}",
  "campaign_dir": "${CAMPAIGN_DIR}",
  "toy_name": "${TOY_NAME}",
  "script_dir": "${SCRIPT_DIR}",
  "python_bin": "${PYTHON_BIN}",
  "python_version": "$(${PYTHON_BIN} --version 2>&1 || true)",
  "script_rel": "${SCRIPT_REL}",
  "output_prefix_rel": "${OUT_PREFIX_REL}",
  "params": {
    "nevents": ${NEVENTS},
    "x": ${X},
    "y": ${Y},
    "qop": ${QOP},
    "qop_phase": ${QOP_PHASE},
    "seed": ${SEED},
    "use_acceptance": ${USE_ACCEPTANCE},
    "acc_type": "${ACC_TYPE}",
    "acc_root_dp": "${ACC_ROOT_DP:-}",
    "acc_hist_pattern": "${ACC_HIST_PATTERN}"
  },
  "artifacts": {
    "command": "${TOY_NAME}_command.sh",
    "run_log": "${TOY_NAME}.log",
    "run_info": "${TOY_NAME}_run_info.txt",
    "manifest": "${TOY_NAME}_manifest.json"
  }
}
EOF

# Optional: Git snapshot (best effort)
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  {
    echo "git_root: $(git rev-parse --show-toplevel)"
    echo "git_branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_status:"
    git status --porcelain
  } > "${CAMPAIGN_DIR}/${TOY_NAME}_git_info.txt" || true

  git diff > "${CAMPAIGN_DIR}/${TOY_NAME}_git_diff.patch" || true
fi

# Acceptance file info (best effort)
if [[ "${USE_ACCEPTANCE}" == "1" && -n "${ACC_ROOT_DP:-}" && -f "${ACC_ROOT_DP}" ]]; then
  {
    echo "acc_file_ls: $(ls -l "${ACC_ROOT_DP}")"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${ACC_ROOT_DP}" | awk '{print "acc_file_sha256: "$1}'
    elif command -v shasum >/dev/null 2>&1; then
      shasum -a 256 "${ACC_ROOT_DP}" | awk '{print "acc_file_sha256: "$1}'
    fi
  } > "${CAMPAIGN_DIR}/${TOY_NAME}_acc_file_info.txt"
fi

# ----------------------------
# Run and capture log
# ----------------------------
echo "[INFO] Campaign dir: ${CAMPAIGN_DIR}"
echo "[INFO] Toy name:      ${TOY_NAME}"
echo "[INFO] Output prefix (rel): ${OUT_PREFIX_REL}"
echo "[INFO] Executing: ${CAMPAIGN_DIR}/${TOY_NAME}_command.sh"

(
  cd "${SCRIPT_DIR}"
  bash "${CAMPAIGN_DIR}/${TOY_NAME}_command.sh"
) &> "${CAMPAIGN_DIR}/${TOY_NAME}.log"

echo "[INFO] Done."
echo "[INFO] Outputs + provenance under:"
echo "       ${CAMPAIGN_DIR}"
