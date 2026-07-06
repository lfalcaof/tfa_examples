from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from d02kspipi_fitcore.runtime.context import FitContext
from d02kspipi_fitcore.runtime.step_executor import StepExecutor
from d02kspipi_fitcore.runtime.batch import BatchLauncher


@dataclass
class DataLoader:
    """Loads data/integration arrays and applies optional Dalitz masks.

    Native v11 implementation of the old step_06_data_loading block.  The
    important compatibility rule is that all symbols produced by the old global
    block are still written into FitContext.namespace with the same names:
    data_np, data_np_dalitz, data_np_mixing, integration_np,
    integration_np_dalitz, integration_np_mixing, data_tf,
    integration_sample, nnorm_dalitz_effective, etc.

    This keeps the still-preserved Dalitz/mixing blocks numerically identical
    while moving data ownership into an explicit class.
    """

    context: FitContext
    executor: StepExecutor

    @property
    def ns(self) -> dict[str, Any]:
        return self.context.namespace

    @staticmethod
    def dalitz_keep_mask(arr, mode: str = "either", lo: float = 0.66, hi: float = 0.87):
        """Build event-level Dalitz mask.

        Column convention:
          arr[:, 0] = m2(KS pi-)
          arr[:, 1] = m2(KS pi+)
          arr[:, 2] = decay time
        """
        m2m = arr[:, 0]
        m2p = arr[:, 1]

        in_m2p = (m2p > lo) & (m2p < hi)
        in_m2m = (m2m > lo) & (m2m < hi)

        if mode == "m2p":
            bad = in_m2p
        elif mode == "m2m":
            bad = in_m2m
        elif mode == "either":
            bad = in_m2p | in_m2m
        elif mode == "both":
            bad = in_m2p & in_m2m
        else:
            raise ValueError("Unknown MASK_MODE: " + str(mode))

        return ~bad

    def load(self) -> None:
        ns = self.ns
        np = ns["np"]
        atfi = ns["atfi"]
        print_section = ns["print_section"]

        print_section("Load data and integration samples")

        defaults = BatchLauncher.configured_default_paths()
        ns["_configured_default_paths"] = BatchLauncher.configured_default_paths
        ns["config"] = defaults

        dataset = ns["DATASET"]
        data_path_override = ns.get("DATA_PATH_OVERRIDE", "")
        integration_path_override = ns.get("INTEGRATION_PATH_OVERRIDE", "")

        if dataset not in defaults:
            raise ValueError(
                f"Dataset '{dataset}' not recognized. Options: {sorted(defaults.keys())}"
            )

        data_path = data_path_override or defaults[dataset]["data"]
        integration_path = integration_path_override or defaults[dataset]["integration"]
        ns["data_path"] = data_path
        ns["integration_path"] = integration_path

        print(f"Using dataset: {dataset}")
        print("Data path:", data_path)
        print("Integration path:", integration_path)

        data_np = np.load(data_path)
        print("Original data shape:", data_np.shape)

        if data_np.shape[1] < 3:
            raise ValueError(
                "The selected data sample does not contain the decay-time column. "
                "Expected at least 3 columns: [m2KsPiMinus, m2KsPiPlus, t]."
            )

        data_np_dalitz = np.ascontiguousarray(data_np[:, :2])
        data_np_mixing = np.ascontiguousarray(data_np[:, :3])

        print("Dalitz-only data shape:", data_np_dalitz.shape)
        print("Dalitz+time data shape:", data_np_mixing.shape)

        integration_np = np.load(integration_path)
        print("Original integration shape:", integration_np.shape)

        if integration_np.shape[1] < 3:
            raise ValueError(
                "The selected integration sample does not contain the decay-time column. "
                "Expected at least 3 columns: [m2KsPiMinus, m2KsPiPlus, t]."
            )

        integration_np_dalitz = np.ascontiguousarray(integration_np[:, :2])
        integration_np_mixing = np.ascontiguousarray(integration_np[:, :3])

        print("Dalitz-only integration shape:", integration_np_dalitz.shape)
        print("Dalitz+time integration shape:", integration_np_mixing.shape)

        ns["dalitz_keep_mask"] = self.dalitz_keep_mask

        if ns["APPLY_DALITZ_MASK"]:
            mask_low = ns["MASK_M2_LOW"]
            mask_high = ns["MASK_M2_HIGH"]
            mask_mode = ns["MASK_MODE"]

            data_col0_in_band = (
                (data_np_mixing[:, 0] > mask_low)
                & (data_np_mixing[:, 0] < mask_high)
            )
            data_col1_in_band = (
                (data_np_mixing[:, 1] > mask_low)
                & (data_np_mixing[:, 1] < mask_high)
            )
            norm_col0_in_band = (
                (integration_np_mixing[:, 0] > mask_low)
                & (integration_np_mixing[:, 0] < mask_high)
            )
            norm_col1_in_band = (
                (integration_np_mixing[:, 1] > mask_low)
                & (integration_np_mixing[:, 1] < mask_high)
            )

            ns.update(
                {
                    "data_col0_in_band": data_col0_in_band,
                    "data_col1_in_band": data_col1_in_band,
                    "norm_col0_in_band": norm_col0_in_band,
                    "norm_col1_in_band": norm_col1_in_band,
                }
            )

            print("")
            print("============================================================")
            print("[DALITZ MASK CHECK] Column-band fractions before masking")
            print("============================================================")
            print("[DALITZ MASK CHECK] Expected convention:")
            print("[DALITZ MASK CHECK]   col0 = m2(KS pi-)")
            print("[DALITZ MASK CHECK]   col1 = m2(KS pi+)")
            print("[DALITZ MASK CHECK] band low :", mask_low)
            print("[DALITZ MASK CHECK] band high:", mask_high)
            print("[DALITZ MASK CHECK] data col0 in band:", np.mean(data_col0_in_band))
            print("[DALITZ MASK CHECK] data col1 in band:", np.mean(data_col1_in_band))
            print("[DALITZ MASK CHECK] norm col0 in band:", np.mean(norm_col0_in_band))
            print("[DALITZ MASK CHECK] norm col1 in band:", np.mean(norm_col1_in_band))
            print("============================================================")
            print("")

            keep_data = self.dalitz_keep_mask(
                data_np_mixing,
                mode=mask_mode,
                lo=mask_low,
                hi=mask_high,
            )
            keep_integration = self.dalitz_keep_mask(
                integration_np_mixing,
                mode=mask_mode,
                lo=mask_low,
                hi=mask_high,
            )

            ns["keep_data"] = keep_data
            ns["keep_integration"] = keep_integration

            print("")
            print("============================================================")
            print("[DALITZ MASK] Applying K*(892) diagnostic mask")
            print("============================================================")
            print("[DALITZ MASK] mode       :", mask_mode)
            print("[DALITZ MASK] m2 low     :", mask_low)
            print("[DALITZ MASK] m2 high    :", mask_high)
            print("[DALITZ MASK] data before:", data_np_mixing.shape)
            print("[DALITZ MASK] norm before:", integration_np_mixing.shape)

            data_np_dalitz = np.ascontiguousarray(data_np_dalitz[keep_data])
            data_np_mixing = np.ascontiguousarray(data_np_mixing[keep_data])
            integration_np_dalitz = np.ascontiguousarray(integration_np_dalitz[keep_integration])
            integration_np_mixing = np.ascontiguousarray(integration_np_mixing[keep_integration])

            print("[DALITZ MASK] data after :", data_np_mixing.shape)
            print("[DALITZ MASK] norm after :", integration_np_mixing.shape)
            print("[DALITZ MASK] data removed fraction:", 1.0 - np.mean(keep_data))
            print("[DALITZ MASK] norm removed fraction :", 1.0 - np.mean(keep_integration))
            print("============================================================")
            print("")
        else:
            print("[DALITZ MASK] Not applied")

        data_tf = atfi.const(data_np_dalitz)

        nnorm_dalitz_effective = min(ns["NNORM"], integration_np_dalitz.shape[0])
        integration_sample = atfi.const(integration_np_dalitz[:nnorm_dalitz_effective])

        print("Final Dalitz-only data sample shape:", data_np_dalitz.shape)
        print("Final Dalitz+time data sample shape:", data_np_mixing.shape)
        print(
            "Final Dalitz-only integration sample shape:",
            integration_np_dalitz[:nnorm_dalitz_effective].shape,
        )
        print("Final Dalitz+time integration sample shape:", integration_np_mixing.shape)

        np.save(ns["ARRAY_DIR"] / "data_np_dalitz.npy", data_np_dalitz[: ns["NFIT"]])
        np.save(
            ns["ARRAY_DIR"] / "data_np_mixing_preview.npy",
            data_np_mixing[: min(ns["NFIT_MIXING"], len(data_np_mixing))],
        )
        np.save(
            ns["ARRAY_DIR"] / "integration_np_dalitz.npy",
            integration_np_dalitz[:nnorm_dalitz_effective],
        )

        ns.update(
            {
                "data_np": data_np,
                "data_np_dalitz": data_np_dalitz,
                "data_np_mixing": data_np_mixing,
                "integration_np": integration_np,
                "integration_np_dalitz": integration_np_dalitz,
                "integration_np_mixing": integration_np_mixing,
                "data_tf": data_tf,
                "nnorm_dalitz_effective": nnorm_dalitz_effective,
                "integration_sample": integration_sample,
            }
        )

    @property
    def data_np_dalitz(self) -> Any:
        return self.context.require("data_np_dalitz")

    @property
    def data_np_mixing(self) -> Any:
        return self.context.require("data_np_mixing")

    @property
    def integration_np_dalitz(self) -> Any:
        return self.context.require("integration_np_dalitz")

    @property
    def integration_np_mixing(self) -> Any:
        return self.context.require("integration_np_mixing")
