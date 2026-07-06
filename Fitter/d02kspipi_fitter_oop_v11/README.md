# D0 -> KS0 pi+ pi- TFA fitter, OOP v11

This directory contains the modular/OOP version of the `D0 -> KS0 pi+ pi-` toy fitter used for Dalitz-plot and time-dependent mixing studies.

The purpose of this refactor is architectural: the validated monolithic/v10 fitter has been reorganized into explicit modules and runner classes while preserving the numerical behavior of the original fit workflow. The remaining known issue is the physics/numerical bias observed in some K*(892)-only mixing toys; that bias is intentionally not hidden or changed by this refactor.

## Status

Current validated checkpoint:

```text
d02kspipi_fitter_oop_v11_validated_diagnostics_clean_xyFloat
```

Validation summary:

```text
1. y-only mixing fit, RUN_NLL_DEBUG=0:
   diagnostics_clean == physics_native_manualfix
   dy  = 0 for 10 seeds
   dey = 0 for 10 seeds

2. y-only mixing fit, RUN_NLL_DEBUG=1:
   seed 1 runs without Traceback/NameError
   native diagnostic helpers are registered
   DALITZ SCORE outputs are produced
   y/ey are identical to the nominal result

3. x,y floating mixing fit:
   10 seeds run successfully
   all mixing_fit_result.json files are produced
```

## Directory layout

```text
run_fitter.py                         main entry point
user_config.py                        user-facing defaults
setup_tfa_gpu.sh                      Brownie/TFA GPU environment setup
validate_oop_syntax.py                syntax/import/block validation
validate_modular_syntax.py            legacy modular syntax validation
MIGRATION_NOTES.md                    migration/validation notes
VALIDATION.md                         reproducibility commands and checks

scripts/
  compare_mixing_results.py           compare seed-by-seed mixing-fit JSONs

d02kspipi_fitcore/
  config.py                           environment-variable configuration
  pipeline.py                         top-level execution sequence

  runtime/                            environment, batch and context setup
  output/                             output folders, logs, JSON, ROOT writing
  data/                               input loading, masks, TensorFlow arrays
  physics/                            masses, amplitudes, Dalitz density model
  fit/                                Dalitz fit, mixing cache/NLL, mixing fit
  diagnostics/                        optional debug/diagnostic helpers
  steps/                              preserved legacy blocks for reference

legacy_reference/
  d02kspipi_fitToys_with_mixing_ACCEPTANCE_COHERENT_v10_withNLL_profile.py
```

The active workflow is routed through the native modules under `d02kspipi_fitcore/`. The `steps/` directory is kept as reference material from the conservative migration and is validated by `validate_oop_syntax.py`, but the nominal workflow no longer depends on executing those preserved blocks directly.

## Environment setup on Brownie

The fitter should be run from a clean terminal using the setup script shipped with this directory.

```bash
cd /user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples/Fitter/d02kspipi_fitter_oop_v11
source setup_tfa_gpu.sh
```

The setup script does the following:

```text
1. clears possible Python/LCG/ROOT contamination variables
2. activates /user/gr1/lhcb/lfalcao/tfa
3. installs lightweight Python dependencies only if missing
4. loads /user/gr1/lhcb/lfalcao/D0toKs0pipi/tfa_examples/setup.sh
5. returns to this fitter directory
6. prints a compact environment summary
```

After sourcing it, check the installation:

```bash
python validate_oop_syntax.py
```

Expected final line:

```text
All v11 OOP fitter modules and preserved blocks compile.
```

## Running a nominal y-only mixing fit

Example used during validation: K*(892)-only toy, generated with `x=0`, `y=-0.006`, fitting only `y_mix`.

```bash
nohup bash -c '
BATCH=1 \
SEED_START=1 \
SEED_END=10 \
USE_GPU=1 \
GPU_ID=0 \
USE_MIXING_AMPLITUDE_CACHE=1 \
TF_MEMORY_GROWTH=1 \
AMP_PARS_MODE=toys \
RUN_DALITZ_FIT=0 \
CREATE_FITTED_SAMPLE=0 \
RUN_MIXING_FIT=1 \
RUN_NLL_DEBUG=0 \
DATASET="NO ACC" \
APPLY_DALITZ_MASK=0 \
USE_MIXING_TMAX_CUT=0 \
MIXING_FLOAT_PARAMS="y_mix" \
X_MIX_INIT=0.0000 \
Y_MIX_INIT=-0.0060 \
NFIT=10000000 \
NNORM=10000000 \
NFIT_MIXING=10000000 \
NNORM_MIXING=10000000 \
OUTPUT_BASE="/data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_yOnly_noTmaxCut_seed1to10_v11_oop" \
DATA_TEMPLATE="/user/gr1/lhcb/lfalcao/D0toKs0pipi/output/only_K892minus_X_0_Y_minus0dot006/Toy_seed{seed}_v9.npy" \
/user/gr1/lhcb/lfalcao/tfa/bin/python run_fitter.py
' > /data01/gr1/lhcb/lfalcao/Fitter_outputs/fit_only_K892minus_X0_Yminus0dot006_yOnly_noTmaxCut_seed1to10_v11_oop.log 2>&1 &
```

## Running with x and y floating

Use the same command but set:

```bash
MIXING_FLOAT_PARAMS="x_mix,y_mix"
```

and choose a different `OUTPUT_BASE`.

## Optional diagnostics

Diagnostics are disabled in the nominal production path:

```bash
RUN_NLL_DEBUG=0
```

To run the NLL/Dalitz-score diagnostics, start with one seed only:

```bash
RUN_NLL_DEBUG=1 \
SEED_START=1 \
SEED_END=1
```

Successful diagnostic registration appears in the log as:

```text
[DIAGNOSTICS] Registered native diagnostic helper functions. [diagnostics_clean]
```

When diagnostics are disabled, the expected log line is:

```text
[DIAGNOSTICS] Optional diagnostics disabled; helper functions not loaded. [diagnostics_clean]
```

## Comparing two runs

Use the helper script:

```bash
python scripts/compare_mixing_results.py \
  --old-base /data01/gr1/lhcb/lfalcao/Fitter_outputs/reference_output_base \
  --new-base /data01/gr1/lhcb/lfalcao/Fitter_outputs/new_output_base \
  --seeds 1-10 \
  --params y_mix
```

For a two-parameter comparison:

```bash
python scripts/compare_mixing_results.py \
  --old-base /data01/gr1/lhcb/lfalcao/Fitter_outputs/reference_output_base \
  --new-base /data01/gr1/lhcb/lfalcao/Fitter_outputs/new_output_base \
  --seeds 1-10 \
  --params x_mix,y_mix
```

## Known physics/numerical issue

The refactor preserves the existing behavior of the fitter. In the K*(892)-only toy generated with `x=0`, `y=-0.006`, the fitter prefers approximately `y ~ -0.0083`. This is a known bias under investigation. It should be studied at the PDF/NLL level using the clean modular version; it is not an architectural/refactoring issue.
