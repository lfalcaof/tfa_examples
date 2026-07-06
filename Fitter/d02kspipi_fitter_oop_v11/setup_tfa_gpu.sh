# Environment setup for the D0 -> KS0 pi+ pi- TFA fitter on Brownie.
#
# Usage from a new terminal:
#   cd /user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples/Fitter/d02kspipi_fitter_oop_v11
#   source setup_tfa_gpu.sh
#
# The script is intended to be sourced, not executed, because it activates
# the Python virtual environment and loads the tfa_examples setup in the
# current shell.

# ------------------------------------------------------------
# Resolve fitter directory
# ------------------------------------------------------------
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    FITTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Fallback for non-bash shells: source this script from the fitter folder.
    FITTER_DIR="$(pwd)"
fi

TFA_VENV="/user/gr1/lhcb/lfalcao/tfa"
TFAEX_DIR="/user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples"

# ------------------------------------------------------------
# Clean possible Python / LCG / ROOT contamination
# ------------------------------------------------------------
unset PYTHONPATH
unset PYTHONHOME
unset LD_LIBRARY_PATH
unset LD_PRELOAD

# ------------------------------------------------------------
# Activate the GPU Python environment
# ------------------------------------------------------------
source "${TFA_VENV}/bin/activate"

# ------------------------------------------------------------
# Install missing lightweight dependencies only if needed
# This does NOT reinstall them every time.
# ------------------------------------------------------------
python - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tabulate": "tabulate",
    "uproot": "uproot",
}

missing = [
    pip_name
    for module_name, pip_name in required.items()
    if importlib.util.find_spec(module_name) is None
]

if missing:
    print("[setup_tfa_gpu] Missing packages:", ", ".join(missing))
    print("[setup_tfa_gpu] Installing them in the active venv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[setup_tfa_gpu] Extra Python dependencies OK.")
PY

# ------------------------------------------------------------
# Load tfa_examples setup
# ------------------------------------------------------------
cd "${TFAEX_DIR}"
. setup.sh

# ------------------------------------------------------------
# Go to fitter directory
# ------------------------------------------------------------
cd "${FITTER_DIR}"

# ------------------------------------------------------------
# Print a compact environment summary
# ------------------------------------------------------------
echo
echo "[setup_tfa_gpu] Environment ready."
echo "[setup_tfa_gpu] python      = $(which python)"
echo "[setup_tfa_gpu] python ver. = $(python --version)"
echo "[setup_tfa_gpu] TFAEX_ROOT  = ${TFAEX_ROOT}"
echo "[setup_tfa_gpu] fitter dir  = ${FITTER_DIR}"
echo "[setup_tfa_gpu] pwd         = $(pwd)"
