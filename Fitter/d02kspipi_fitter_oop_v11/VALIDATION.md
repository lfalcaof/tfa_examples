# Validation and final pre-git checklist

This file records the final checks recommended before pushing the v11 OOP fitter to git.

## 1. Environment setup

From a fresh Brownie terminal:

```bash
cd /user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples/Fitter/d02kspipi_fitter_oop_v11
source setup_tfa_gpu.sh
python validate_oop_syntax.py
```

Expected final line:

```text
All v11 OOP fitter modules and preserved blocks compile.
```

## 2. Check that diagnostics-clean markers are present and old hacks are absent

```bash
grep -R "diagnostics_clean\|builtins.MIXING_NLL_CHUNK\|manual-builtins-fix\|chunk_size=MIXING_NLL_CHUNK" -n \
  run_fitter.py d02kspipi_fitcore/diagnostics d02kspipi_fitcore/fit d02kspipi_fitcore/runtime d02kspipi_fitcore/pipeline.py
```

Expected:

```text
[diagnostics_clean] appears in diagnostics_runner.py
builtins.MIXING_NLL_CHUNK does not appear
manual-builtins-fix does not appear
chunk_size=MIXING_NLL_CHUNK does not appear
```

## 3. Final y-only comparison

Reference base:

```text
/data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_yOnly_noTmaxCut_seed1to10_v11_oop_physics_native_manualfix
```

New base:

```text
/data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_yOnly_noTmaxCut_seed1to10_v11_oop_diagnostics_clean
```

Comparison command:

```bash
python scripts/compare_mixing_results.py \
  --old-base /data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_yOnly_noTmaxCut_seed1to10_v11_oop_physics_native_manualfix \
  --new-base /data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_yOnly_noTmaxCut_seed1to10_v11_oop_diagnostics_clean \
  --seeds 1-10 \
  --params y_mix
```

Expected:

```text
dy  = 0 for all seeds
dey = 0 for all seeds
```

## 4. Final diagnostic smoke test

For `RUN_NLL_DEBUG=1`, one seed is enough for the smoke test.

Expected log lines:

```text
[DIAGNOSTICS] Registered native diagnostic helper functions. [diagnostics_clean]
[DALITZ SCORE] Saved arrays: .../dalitz_score_x_profile_zero.npz
[DALITZ SCORE] Saved plots in: ...
```

Expected fit comparison against the nominal seed 1:

```text
dy  = 0
dey = 0
```

## 5. Final x,y-floating smoke test

The final x,y-floating test should produce all ten JSON files:

```bash
BASE="/data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_xyFloat_noTmaxCut_seed1to10_v11_oop_diagnostics_clean"

for seed in {1..10}; do
  f="${BASE}_seed${seed}/json/mixing_fit_result.json"
  if [ -f "$f" ]; then
    echo "seed ${seed}: OK"
  else
    echo "seed ${seed}: MISSING $f"
  fi
done
```

## 6. Clean local tree before git

From the fitter directory:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
```

Then inspect:

```bash
git status --short
find . -maxdepth 3 -type f | sort
```

Do not commit generated outputs, ROOT files, logs, large `.npy/.npz` files, or seed-specific output folders.
