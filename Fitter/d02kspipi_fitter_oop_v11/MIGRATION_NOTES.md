# Migration notes: monolithic/v10 -> OOP v11

## Goal

The goal of v11 is to make the fitter maintainable and user-friendly without changing the validated physics/PDF behavior. The monolithic workflow was split into explicit modules/classes, and every migration step was checked against the previous validated checkpoint.

This is a conservative refactor, not a physics rewrite.

## Active execution sequence

The current pipeline is owned by `FitterPipeline`:

```text
1. RuntimeEnvironment.configure()
2. RuntimeEnvironment.run_batch_if_requested()
3. OutputManager.setup()
4. PhysicsModel.setup_framework_and_constants()
5. PhysicsModel.build_density_model()
6. DataLoader.load()
7. DalitzFitRunner.run()
8. PhysicsModel.build_complex_amplitude_model_and_projection()
9. MixingFitRunner.prepare_samples()
10. MixingFitRunner.build_cache_and_nll_tools()
11. DiagnosticsRunner.register_functions()
12. MixingFitRunner.run_fit()
13. DiagnosticsRunner.run_optional()
14. OutputManager.write_final_outputs()
```

## Native/OOP components

The active workflow now uses native modules/classes for:

```text
configuration/runtime
batch launching
output management
data loading and optional Dalitz masks
physics framework/constants
Dalitz density model
Dalitz fit control flow
complex-amplitude projection
mixing sample preparation
mixing cache/NLL utilities
mixing fit control flow
optional diagnostics
final output summaries
```

The corresponding code lives mainly in:

```text
d02kspipi_fitcore/config.py
d02kspipi_fitcore/pipeline.py
d02kspipi_fitcore/runtime/
d02kspipi_fitcore/output/
d02kspipi_fitcore/data/
d02kspipi_fitcore/physics/
d02kspipi_fitcore/fit/
d02kspipi_fitcore/diagnostics/
```

## Why `steps/` still exists

The `d02kspipi_fitcore/steps/` directory is retained as reference material from the migration. It allows direct comparison with the legacy block organization and is still included in syntax/block validation.

The nominal workflow is routed through the native OOP modules and no longer relies on executing the preserved `steps/` blocks as the primary implementation.

## Final validated checkpoint

Final checkpoint name used in the Brownie working area:

```text
d02kspipi_fitter_oop_v11_validated_diagnostics_clean_xyFloat
```

Validation summary:

```text
1. y-only, RUN_NLL_DEBUG=0:
   diagnostics_clean == physics_native_manualfix
   dy  = 0 for seeds 1..10
   dey = 0 for seeds 1..10

2. y-only, RUN_NLL_DEBUG=1:
   seed 1 completes successfully
   native diagnostics are registered
   DALITZ SCORE arrays/plots are produced
   y/ey are identical to the nominal result

3. x,y float:
   seeds 1..10 complete successfully
   all mixing_fit_result.json files are produced
```

## Important interpretation

The refactor has not changed the observed K*(892)-only mixing bias. That is intentional: the purpose of this migration was to preserve behavior exactly while making the fitter easier to understand and debug.

The next work item is therefore not another structural refactor, but a physics/numerical investigation of the bias using the clean v11 codebase.
