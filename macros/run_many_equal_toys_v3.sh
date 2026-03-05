#!/usr/bin/env bash
set -euo pipefail

# ============================================
# run_many_equal_toys_v3.sh
# - Runs d02kspipi_toys_v3.py N times
# - Each run uses a different --seed
# - All outputs go to ONE folder under ../../output/
# - Output name per toy: Toy_seedX_v3
#
# v3 additions:
# - Analytic time acceptance (RooAcceptanceDT-like) via:
#     --use_time_acceptance
#     --ta_x0 --ta_a --ta_b --ta_beta --ta_n --ta_m
# - Parameters depend on ACC_TYPE (trigger line)
#
# NEW:
# - Exact dt-fit-PDF mode via:
#     --use_time_fitpdf_exact
#   (mutually exclusive with --use_time_acceptance)
#
# NEW (this patch):
# - Optional proposal-compensation in t via:
#     --compensate_time_proposal_exp
#   (recommended mainly with --use_time_fitpdf_exact when DecayTimePhaseSpace proposes exp(-t/tau))
# ============================================
# Run example:
#   nohup ./run_many_equal_toys_v3.sh 1000 > Equal_1000toys.log 2>&1 & disown

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------
N="${1:-5}"                             # number of toys/runs (seeds 1..N)

USE_ACCEPTANCE="${USE_ACCEPTANCE:-1}"   # 1=yes (DP×time maps), 0=no

# Choose exactly ONE time option (defaults keep backward-compatibility)
USE_TIME_ACCEPTANCE="${USE_TIME_ACCEPTANCE:-0}"        # 1=yes (MODE B), 0=no
USE_TIME_FITPDF_EXACT="${USE_TIME_FITPDF_EXACT:-1}"    # 1=yes (fitPDF exact), 0=no

# Method to compensate toggle residual exp(-t) in toys.
COMPENSATE_TIME_PROPOSAL_EXP="${COMPENSATE_TIME_PROPOSAL_EXP:-1}" # 1=yes, 0=no

# Allowed: selTwoTracks | selD2Kshh | selD2Kshh_TwoTracks
ACC_TYPE="${ACC_TYPE:-selD2Kshh_TwoTracks}"


PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_REL="${SCRIPT_REL:-d02kspipi_toys_v3.py}"

NEVENTS="${NEVENTS:-2000000}"
X="${X:-0.004}"
Y="${Y:-0.006}"
QOP="${QOP:-1.0}"
QOP_PHASE="${QOP_PHASE:-0.0}"

# ------------------------------------------------------------------------------------

# ----------------------------
# Guards (time options)
# ----------------------------
if [[ "${USE_TIME_ACCEPTANCE}" == "1" && "${USE_TIME_FITPDF_EXACT}" == "1" ]]; then
  echo "[ERROR] Choose only ONE time mode:" >&2
  echo "        USE_TIME_ACCEPTANCE=1  OR  USE_TIME_FITPDF_EXACT=1" >&2
  exit 10
fi

# (optional) only meaningful when fitpdf_exact is ON, but we don't forbid other combos
if [[ "${COMPENSATE_TIME_PROPOSAL_EXP}" == "1" && "${USE_TIME_FITPDF_EXACT}" != "1" ]]; then
  echo "[WARN] COMPENSATE_TIME_PROPOSAL_EXP=1 is typically intended for USE_TIME_FITPDF_EXACT=1." >&2
fi

# ----------------------------
# Tags / campaign name
# ----------------------------
ACC_TAG="NoACC"
if [[ "$USE_ACCEPTANCE" -eq 1 ]]; then
  ACC_TAG="WithACC_${ACC_TYPE}"
fi

TA_TAG="NoTA"
TFIT_TAG="NoTFit"

if [[ "$USE_TIME_ACCEPTANCE" -eq 1 ]]; then
  TA_TAG="WithTA_${ACC_TYPE}"
fi

if [[ "$USE_TIME_FITPDF_EXACT" -eq 1 ]]; then
  TFIT_TAG="WithTFit_${ACC_TYPE}"
fi

COMP_TAG="NoComp"
if [[ "$COMPENSATE_TIME_PROPOSAL_EXP" -eq 1 ]]; then
  COMP_TAG="WithComp"
fi

echo "[INFO] Toys generated: ${ACC_TAG} + ${TA_TAG} + ${TFIT_TAG} + ${COMP_TAG}"
CAMPAIGN="${CAMPAIGN:-Equal_${N}toys_v3_${ACC_TAG}_${TA_TAG}_${TFIT_TAG}_${COMP_TAG}}"

# Must be Python format-style if python uses hist_pattern.format(i)
ACC_HIST_PATTERN="${ACC_HIST_PATTERN:-accDP_tbin_{:02d}}"

# Base output directory (relative to this script location)
OUTPUT_BASE_DIR_REL="${OUTPUT_BASE_DIR_REL:-../../output}"

# ----------------------------
# Anchor to script dir
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ----------------------------
# Select DP×time acceptance ROOT file (if enabled)
# ----------------------------
ACC_ROOT_DP=""
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
# Analytic time-acceptance parameters per trigger (if enabled)
# (values taken from your RooFit results)
# ----------------------------
TA_X0="${TA_X0:-}"
TA_A="${TA_A:-}"
TA_B="${TA_B:-}"
TA_BETA="${TA_BETA:-}"
TA_N="${TA_N:-}"
TA_M="${TA_M:-}"

# NOTE: both time modes need the same RooAcceptanceDT params
if [[ "${USE_TIME_ACCEPTANCE}" == "1" || "${USE_TIME_FITPDF_EXACT}" == "1" ]]; then
  # If not overridden via env, set defaults from RooFit per ACC_TYPE
  if [[ -z "${TA_X0}" || -z "${TA_A}" || -z "${TA_B}" || -z "${TA_BETA}" || -z "${TA_N}" || -z "${TA_M}" ]]; then
    case "${ACC_TYPE}" in
      selD2Kshh)
        TA_X0="${TA_X0:-4.8266e-01}"
        TA_A="${TA_A:-8.1952e-01}"
        TA_B="${TA_B:-1.7977e-01}"
        TA_BETA="${TA_BETA:-1.1499e+00}"
        TA_N="${TA_N:-3.5189e+00}"
        TA_M="${TA_M:-1.0322e-01}"
        ;;
      selTwoTracks)
        TA_X0="${TA_X0:-3.1942e-01}"
        TA_A="${TA_A:-3.3439e+00}"
        TA_B="${TA_B:-3.9281e-01}"
        TA_BETA="${TA_BETA:-1.1911e+00}"
        TA_N="${TA_N:-2.9968e+00}"
        TA_M="${TA_M:-9.1052e-01}"
        ;;
      selD2Kshh_TwoTracks)
        TA_X0="${TA_X0:-3.5391e-01}"
        TA_A="${TA_A:-3.2099e+00}"
        TA_B="${TA_B:-1.9144e-01}"
        TA_BETA="${TA_BETA:-1.2719e+00}"
        TA_N="${TA_N:-3.6467e+00}"
        TA_M="${TA_M:-6.4542e-01}"
        ;;
      *)
        echo "[ERROR] Unknown ACC_TYPE='${ACC_TYPE}' for time-acceptance parameter mapping" >&2
        exit 5
        ;;
    esac
  fi
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

use_acceptance (DP×time maps): ${USE_ACCEPTANCE}
acc_type: ${ACC_TYPE}
acc_root_dp: ${ACC_ROOT_DP}
acc_hist_pattern: ${ACC_HIST_PATTERN}

use_time_acceptance (MODE B): ${USE_TIME_ACCEPTANCE}
use_time_fitpdf_exact: ${USE_TIME_FITPDF_EXACT}
compensate_time_proposal_exp: ${COMPENSATE_TIME_PROPOSAL_EXP}

ta_x0: ${TA_X0}
ta_a: ${TA_A}
ta_b: ${TA_B}
ta_beta: ${TA_BETA}
ta_n: ${TA_N}
ta_m: ${TA_M}
EOF

# Sanity checks
if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
  if [[ "${ACC_HIST_PATTERN}" == *"}"* && "${ACC_HIST_PATTERN}" != *"{"* ]]; then
    echo "[ERROR] ACC_HIST_PATTERN seems to contain '}' but no '{': ${ACC_HIST_PATTERN}" >&2
    echo "        Use something like: accDP_tbin_{:02d}" >&2
    exit 2
  fi

  if [[ -z "${ACC_ROOT_DP}" ]]; then
    echo "[ERROR] ACC_ROOT_DP is empty but USE_ACCEPTANCE=1" >&2
    exit 4
  fi

  if [[ ! -f "${SCRIPT_DIR}/${ACC_ROOT_DP}" && ! -f "${ACC_ROOT_DP}" ]]; then
    echo "[ERROR] ACC_ROOT_DP not found: ${ACC_ROOT_DP}" >&2
    echo "        Checked: '${SCRIPT_DIR}/${ACC_ROOT_DP}' and '${ACC_ROOT_DP}'" >&2
    exit 4
  fi
fi

if [[ "${USE_TIME_ACCEPTANCE}" == "1" || "${USE_TIME_FITPDF_EXACT}" == "1" ]]; then
  for v in TA_X0 TA_A TA_B TA_BETA TA_N TA_M; do
    if [[ -z "${!v}" ]]; then
      echo "[ERROR] ${v} is empty but a time-mode is enabled" >&2
      exit 6
    fi
  done
fi

echo "[INFO] Campaign dir: ${CAMPAIGN_DIR}"
echo "[INFO] Running ${N} toys..."

for ((seed=1; seed<=N; seed++)); do
  toy_name="Toy_seed${seed}_v3"
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

  # Time mode: MODE B (internal time_acceptance_from_fitpdf)
  if [[ "${USE_TIME_ACCEPTANCE}" == "1" ]]; then
    cmd+=(
      --use_time_acceptance
      --ta_x0 "${TA_X0}"
      --ta_a "${TA_A}"
      --ta_b "${TA_B}"
      --ta_beta "${TA_BETA}"
      --ta_n "${TA_N}"
      --ta_m "${TA_M}"
    )
  fi

  # Time mode: fitPDF exact (psi -> 1 and RooAcceptanceDT-like used as full PDF)
  if [[ "${USE_TIME_FITPDF_EXACT}" == "1" ]]; then
    cmd+=(
      --use_time_fitpdf_exact
      --ta_x0 "${TA_X0}"
      --ta_a "${TA_A}"
      --ta_b "${TA_B}"
      --ta_beta "${TA_BETA}"
      --ta_n "${TA_N}"
      --ta_m "${TA_M}"
    )

    # NEW: optional proposal compensation (minimal addition)
    if [[ "${COMPENSATE_TIME_PROPOSAL_EXP}" == "1" ]]; then
      cmd+=( --compensate_time_proposal_exp )
    fi
  fi

  # DP×time maps acceptance
  if [[ "${USE_ACCEPTANCE}" == "1" ]]; then
    cmd+=( --use_acceptance --acc_root_dp "${ACC_ROOT_DP}" --acc_hist_pattern "${ACC_HIST_PATTERN}" )
  fi

  echo "[INFO] (${seed}/${N}) ${toy_name}"
  printf '%q ' "${cmd[@]}" > "${CAMPAIGN_DIR}/${toy_name}_command.sh"
  echo >> "${CAMPAIGN_DIR}/${toy_name}_command.sh"
  chmod +x "${CAMPAIGN_DIR}/${toy_name}_command.sh"

  (
    cd "${SCRIPT_DIR}"
    "${CAMPAIGN_DIR}/${toy_name}_command.sh"
  ) &> "${log_file}"
done

echo "[INFO] Done. All logs and commands are in:"
echo "       ${CAMPAIGN_DIR}"