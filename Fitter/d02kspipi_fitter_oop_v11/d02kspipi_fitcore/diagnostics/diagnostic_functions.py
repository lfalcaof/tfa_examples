# Native diagnostic helper functions for v11.
#
# This module contains the former step_11 diagnostic functions as real Python
# definitions. It is intentionally loaded by DiagnosticsRunner with the active
# FitContext namespace pre-populated, so symbols such as np/tf/atfi and
# mixing_density_cached_amplitudes preserve the same values they had in the
# validated v10/v11 execution flow. Chunk-size defaults are resolved at call
# time from this controlled namespace, not at function-definition time.


def _mixing_nll_chunk():
    return int(globals().get("MIXING_NLL_CHUNK", 250000))

def _mixing_cache_chunk():
    return int(globals().get("MIXING_CACHE_CHUNK", 250000))

# ============================================================
# Fixed-point NLL diagnostic utilities
# ============================================================

def make_fixed_mixing_pars(x, y, qp_abs=1.0, qp_phi=0.0):
    """Build fixed mixing parameters for direct NLL evaluation."""

    return {
        "x_mix": atfi.const(float(x)),
        "y_mix": atfi.const(float(y)),
        "qp_abs": atfi.const(float(qp_abs)),
        "qp_phi": atfi.const(float(qp_phi)),
    }


def evaluate_cached_nll_terms(
    data_cache,
    norm_cache,
    x,
    y,
    qp_abs=1.0,
    qp_phi=0.0,
    chunk_size=None,
):
    """Evaluate cached NLL terms at fixed x,y without running Minuit.

    This uses exactly the same cached density as the fit:
        mixing_density_cached_amplitudes(...)

    It returns separately:
        data_term = -sum log(pdf_data)
        norm_term = N_data * log(<pdf_norm>)
        total_nll = data_term + norm_term
    """
    if chunk_size is None:
        chunk_size = _mixing_nll_chunk()


    pars = make_fixed_mixing_pars(x, y, qp_abs=qp_abs, qp_phi=qp_phi)

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    data_log_sum = 0.0
    norm_sum = 0.0

    data_min_pdf = +1.0e300
    data_max_pdf = -1.0e300
    norm_min_pdf = +1.0e300
    norm_max_pdf = -1.0e300

    data_bad_pdf = 0
    norm_bad_pdf = 0

    # ------------------------------------------------------------
    # Data term: -sum log(pdf_data)
    # ------------------------------------------------------------
    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        pdf = mixing_density_cached_amplitudes(
            tf.constant(data_cache["A"][start:stop]),
            tf.constant(data_cache["Abar"][start:stop]),
            atfi.const(data_cache["t"][start:stop]),
            pars["x_mix"],
            pars["y_mix"],
            pars["qp_abs"],
            pars["qp_phi"],
        )

        pdf_np = pdf.numpy()

        data_min_pdf = min(data_min_pdf, float(np.min(pdf_np)))
        data_max_pdf = max(data_max_pdf, float(np.max(pdf_np)))
        data_bad_pdf += int(np.sum(pdf_np <= 0.0))

        data_log_sum += float(tf.reduce_sum(tf.math.log(pdf)).numpy())

    # ------------------------------------------------------------
    # Normalization term: N_data * log(mean(pdf_norm))
    # ------------------------------------------------------------
    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        pdf = mixing_density_cached_amplitudes(
            tf.constant(norm_cache["A"][start:stop]),
            tf.constant(norm_cache["Abar"][start:stop]),
            atfi.const(norm_cache["t"][start:stop]),
            pars["x_mix"],
            pars["y_mix"],
            pars["qp_abs"],
            pars["qp_phi"],
        )

        pdf_np = pdf.numpy()

        norm_min_pdf = min(norm_min_pdf, float(np.min(pdf_np)))
        norm_max_pdf = max(norm_max_pdf, float(np.max(pdf_np)))
        norm_bad_pdf += int(np.sum(pdf_np <= 0.0))

        norm_sum += float(tf.reduce_sum(pdf).numpy())

    norm_int = norm_sum / float(n_norm_total)
    norm_int = np.clip(norm_int, 1.0e-300, 1.0e300)

    data_term = -data_log_sum
    norm_term = float(n_data_total) * np.log(norm_int)
    total_nll = data_term + norm_term

    return {
        "x": float(x),
        "y": float(y),
        "data_term": data_term,
        "norm_term": norm_term,
        "total_nll": total_nll,
        "norm_int": norm_int,
        "data_min_pdf": data_min_pdf,
        "data_max_pdf": data_max_pdf,
        "norm_min_pdf": norm_min_pdf,
        "norm_max_pdf": norm_max_pdf,
        "data_bad_pdf": data_bad_pdf,
        "norm_bad_pdf": norm_bad_pdf,
    }




def finite_difference_x_diagnostics(
    data_cache,
    norm_cache,
    y_fixed=0.0,
    x0=0.0,
    eps=1.0e-4,
    chunk_size=None,
):
    """
    Compute finite-difference gradient and curvature in x at fixed y.

    g_x  = dNLL/dx
    H_xx = d2NLL/dx2

    The local predicted minimum, at fixed y, is approximately:

        x_pred = x0 - g_x / H_xx
    """
    if chunk_size is None:
        chunk_size = _mixing_nll_chunk()


    nll_m = evaluate_cached_nll_terms(
        data_cache=data_cache,
        norm_cache=norm_cache,
        x=x0 - eps,
        y=y_fixed,
        chunk_size=chunk_size,
    )["total_nll"]

    nll_0 = evaluate_cached_nll_terms(
        data_cache=data_cache,
        norm_cache=norm_cache,
        x=x0,
        y=y_fixed,
        chunk_size=chunk_size,
    )["total_nll"]

    nll_p = evaluate_cached_nll_terms(
        data_cache=data_cache,
        norm_cache=norm_cache,
        x=x0 + eps,
        y=y_fixed,
        chunk_size=chunk_size,
    )["total_nll"]

    grad_x = (nll_p - nll_m) / (2.0 * eps)
    hess_xx = (nll_p - 2.0 * nll_0 + nll_m) / (eps * eps)

    x_pred = np.nan
    if hess_xx != 0.0:
        x_pred = x0 - grad_x / hess_xx

    print()
    print("=" * 100)
    print("Finite-difference x diagnostic")
    print("=" * 100)
    print(f"x0       = {x0:.8f}")
    print(f"y_fixed  = {y_fixed:.8f}")
    print(f"eps      = {eps:.8e}")
    print(f"NLL(x-e) = {nll_m:.10f}")
    print(f"NLL(x0)  = {nll_0:.10f}")
    print(f"NLL(x+e) = {nll_p:.10f}")
    print(f"grad_x   = {grad_x:.10e}")
    print(f"hess_xx  = {hess_xx:.10e}")
    print(f"x_pred   = {x_pred:.10e}")
    print("=" * 100)

    return {
        "x0": x0,
        "y_fixed": y_fixed,
        "eps": eps,
        "nll_minus": nll_m,
        "nll_zero": nll_0,
        "nll_plus": nll_p,
        "grad_x": grad_x,
        "hess_xx": hess_xx,
        "x_pred": x_pred,
    }




def run_fixed_point_nll_check(data_cache, norm_cache):
    """Run the first fixed-point NLL checks for the zero-mixing toy."""

    test_points = [
        ("truth",       0.000000,  0.000000),
        ("fit_result", -0.005482, -0.000917),
        ("x_offset",   -0.005600,  0.000000),
        ("opposite_x", +0.005600,  0.000000),
        ("y_offset",    0.000000, -0.000917),
    ]

    rows = []

    print()
    print("=" * 80)
    print("Fixed-point NLL check")
    print("=" * 80)

    for label, x, y in test_points:
        print(f"[NLL CHECK] Evaluating {label}: x = {x:.6f}, y = {y:.6f}")

        out = evaluate_cached_nll_terms(
            data_cache=data_cache,
            norm_cache=norm_cache,
            x=x,
            y=y,
        )

        out["label"] = label
        rows.append(out)

    min_total = min(row["total_nll"] for row in rows)
    min_data = min(row["data_term"] for row in rows)
    min_norm = min(row["norm_term"] for row in rows)

    print()
    print("=" * 120)
    print("NLL term decomposition")
    print("=" * 120)
    print(
        f"{'label':<15} {'x':>11} {'y':>11} "
        f"{'DeltaNLL':>15} {'DeltaData':>15} {'DeltaNorm':>15} "
        f"{'norm_int':>15} {'data_min_pdf':>15} {'norm_min_pdf':>15} "
        f"{'data_bad':>10} {'norm_bad':>10}"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['label']:<15} "
            f"{row['x']:>11.6f} "
            f"{row['y']:>11.6f} "
            f"{row['total_nll'] - min_total:>15.6f} "
            f"{row['data_term'] - min_data:>15.6f} "
            f"{row['norm_term'] - min_norm:>15.6f} "
            f"{row['norm_int']:>15.8e} "
            f"{row['data_min_pdf']:>15.8e} "
            f"{row['norm_min_pdf']:>15.8e} "
            f"{row['data_bad_pdf']:>10d} "
            f"{row['norm_bad_pdf']:>10d}"
        )

    print("=" * 120)

    return rows
    

def run_x_scan_nll_check(
    data_cache,
    norm_cache,
    y_fixed=0.0,
    x_min=-0.010,
    x_max=0.002,
    n_points=49,
    chunk_size=None,
):
    """Scan the cached NLL as a function of x with y fixed."""
    if chunk_size is None:
        chunk_size = _mixing_nll_chunk()


    print()
    print("=" * 100)
    print("DEBUG: 1D NLL scan in x")
    print("=" * 100)
    print(f"[X SCAN] y fixed = {y_fixed}")
    print(f"[X SCAN] range   = [{x_min}, {x_max}]")
    print(f"[X SCAN] points  = {n_points}")

    x_values = np.linspace(x_min, x_max, n_points)

    rows = []

    for x_value in x_values:
        print(f"[X SCAN] Evaluating x = {x_value:.8f}, y = {y_fixed:.8f}")

        out = evaluate_cached_nll_terms(
            data_cache=data_cache,
            norm_cache=norm_cache,
            x=float(x_value),
            y=float(y_fixed),
            qp_abs=1.0,
            qp_phi=0.0,
            chunk_size=chunk_size,
        )

        rows.append(out)

    min_total = min(row["total_nll"] for row in rows)
    min_row = min(rows, key=lambda row: row["total_nll"])

    print()
    print("=" * 100)
    print("1D x scan result")
    print("=" * 100)
    print(f"[X SCAN] minimum x = {min_row['x']:.8f}")
    print(f"[X SCAN] minimum y = {min_row['y']:.8f}")
    print(f"[X SCAN] minimum NLL = {min_row['total_nll']:.8f}")

    print()
    print(f"{'x':>12} {'y':>12} {'NLL':>20} {'DeltaNLL':>15}")
    print("-" * 65)

    for row in rows:
        print(
            f"{row['x']:>12.8f} "
            f"{row['y']:>12.8f} "
            f"{row['total_nll']:>20.6f} "
            f"{row['total_nll'] - min_total:>15.6f}"
        )

    print("=" * 100)

    return rows



def run_score_profile_in_dalitz_at_zero(
    data_np,
    data_cache,
    norm_np,
    norm_cache,
    eps=1.0e-4,
    n_bins_x=50,
    n_bins_y=50,
    chunk_size=None,
    output_dir=None,
):
    """Map the local x-score contribution in the Dalitz plot at x=0, y=0.

    The local contribution to the NLL gradient is

        grad_bin = - sum_data_bin score_x
                   + N_data * sum_norm_bin[pdf0 * score_x] / sum_norm_all[pdf0]

    Positive grad_bin means that bin pushes the NLL to decrease for x < 0.
    Negative grad_bin means that bin pushes the NLL to decrease for x > 0.
    """
    if chunk_size is None:
        chunk_size = _mixing_nll_chunk()


    print()
    print("=" * 100)
    print("DEBUG: score_x profile in Dalitz plot at x=0, y=0")
    print("=" * 100)

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    data_xy = np.asarray(data_np[:n_data_total, 0:2], dtype=np.float64)
    norm_xy = np.asarray(norm_np[:n_norm_total, 0:2], dtype=np.float64)

    x_min = min(float(np.min(data_xy[:, 0])), float(np.min(norm_xy[:, 0])))
    x_max = max(float(np.max(data_xy[:, 0])), float(np.max(norm_xy[:, 0])))
    y_min = min(float(np.min(data_xy[:, 1])), float(np.min(norm_xy[:, 1])))
    y_max = max(float(np.max(data_xy[:, 1])), float(np.max(norm_xy[:, 1])))

    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)

    print(f"[DALITZ SCORE] x range = [{x_min:.8f}, {x_max:.8f}]")
    print(f"[DALITZ SCORE] y range = [{y_min:.8f}, {y_max:.8f}]")
    print(f"[DALITZ SCORE] bins    = {n_bins_x} x {n_bins_y}")

    data_count = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    data_score_sum = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    norm_weight_sum = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    norm_weighted_score_sum = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)

    # ------------------------------------------------------------
    # Data part
    # ------------------------------------------------------------
    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        A = tf.constant(data_cache["A"][start:stop])
        Abar = tf.constant(data_cache["Abar"][start:stop])
        t = atfi.const(data_cache["t"][start:stop])

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        score_x_np = score_x.numpy()

        xy_chunk = data_xy[start:stop]
        ix = np.searchsorted(x_edges, xy_chunk[:, 0], side="right") - 1
        iy = np.searchsorted(y_edges, xy_chunk[:, 1], side="right") - 1

        valid = (ix >= 0) & (ix < n_bins_x) & (iy >= 0) & (iy < n_bins_y)

        np.add.at(data_count, (ix[valid], iy[valid]), 1.0)
        np.add.at(data_score_sum, (ix[valid], iy[valid]), score_x_np[valid])

    # ------------------------------------------------------------
    # Normalization expectation
    # ------------------------------------------------------------
    total_norm_weight = 0.0

    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        A = tf.constant(norm_cache["A"][start:stop])
        Abar = tf.constant(norm_cache["Abar"][start:stop])
        t = atfi.const(norm_cache["t"][start:stop])

        pdf_0 = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        pdf_0_np = pdf_0.numpy()
        score_x_np = score_x.numpy()

        total_norm_weight += float(np.sum(pdf_0_np))

        xy_chunk = norm_xy[start:stop]
        ix = np.searchsorted(x_edges, xy_chunk[:, 0], side="right") - 1
        iy = np.searchsorted(y_edges, xy_chunk[:, 1], side="right") - 1

        valid = (ix >= 0) & (ix < n_bins_x) & (iy >= 0) & (iy < n_bins_y)

        np.add.at(norm_weight_sum, (ix[valid], iy[valid]), pdf_0_np[valid])
        np.add.at(
            norm_weighted_score_sum,
            (ix[valid], iy[valid]),
            pdf_0_np[valid] * score_x_np[valid],
        )

    # ------------------------------------------------------------
    # Local normalized NLL gradient contribution
    # ------------------------------------------------------------
    local_grad_x = (
        -data_score_sum
        + float(n_data_total) * norm_weighted_score_sum / total_norm_weight
    )

    data_frac = data_count / float(np.sum(data_count))
    norm_frac = norm_weight_sum / float(np.sum(norm_weight_sum))

    data_mean_score_x = data_score_sum / np.maximum(data_count, 1.0)
    norm_mean_score_x = norm_weighted_score_sum / np.maximum(norm_weight_sum, 1.0)

    score_diff = data_mean_score_x - norm_mean_score_x
    frac_diff = data_frac - norm_frac

    total_grad_x = float(np.sum(local_grad_x))

    print()
    print("=" * 100)
    print("Dalitz score_x profile summary")
    print("=" * 100)
    print(f"[DALITZ SCORE] total local grad_x = {total_grad_x:.10e}")
    print("[DALITZ SCORE] positive local grad_x pushes the fit toward x < 0")
    print("[DALITZ SCORE] negative local grad_x pushes the fit toward x > 0")

    flat_grad = local_grad_x.reshape(-1)
    top_pos = np.argsort(flat_grad)[-15:][::-1]
    top_neg = np.argsort(flat_grad)[:15]

    print()
    print("Top positive bins: strongest pull toward x < 0")
    print("-" * 110)
    print(
        f"{'rank':>4} {'ix':>4} {'iy':>4} "
        f"{'m2p_low':>11} {'m2p_high':>11} "
        f"{'m2m_low':>11} {'m2m_high':>11} "
        f"{'grad_x':>15} {'data_frac':>12} {'norm_frac':>12} {'score_diff':>15}"
    )

    for rank, idx in enumerate(top_pos, start=1):
        ix = idx // n_bins_y
        iy = idx % n_bins_y

        print(
            f"{rank:>4d} {ix:>4d} {iy:>4d} "
            f"{x_edges[ix]:>11.6f} {x_edges[ix+1]:>11.6f} "
            f"{y_edges[iy]:>11.6f} {y_edges[iy+1]:>11.6f} "
            f"{local_grad_x[ix, iy]:>15.6e} "
            f"{data_frac[ix, iy]:>12.6e} "
            f"{norm_frac[ix, iy]:>12.6e} "
            f"{score_diff[ix, iy]:>15.6e}"
        )

    print()
    print("Top negative bins: strongest pull toward x > 0")
    print("-" * 110)
    print(
        f"{'rank':>4} {'ix':>4} {'iy':>4} "
        f"{'m2p_low':>11} {'m2p_high':>11} "
        f"{'m2m_low':>11} {'m2m_high':>11} "
        f"{'grad_x':>15} {'data_frac':>12} {'norm_frac':>12} {'score_diff':>15}"
    )

    for rank, idx in enumerate(top_neg, start=1):
        ix = idx // n_bins_y
        iy = idx % n_bins_y

        print(
            f"{rank:>4d} {ix:>4d} {iy:>4d} "
            f"{x_edges[ix]:>11.6f} {x_edges[ix+1]:>11.6f} "
            f"{y_edges[iy]:>11.6f} {y_edges[iy+1]:>11.6f} "
            f"{local_grad_x[ix, iy]:>15.6e} "
            f"{data_frac[ix, iy]:>12.6e} "
            f"{norm_frac[ix, iy]:>12.6e} "
            f"{score_diff[ix, iy]:>15.6e}"
        )

    print("=" * 100)

    # ------------------------------------------------------------
    # Save arrays and optional plots
    # ------------------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_npz = output_dir / "dalitz_score_x_profile_zero.npz"

        np.savez(
            out_npz,
            x_edges=x_edges,
            y_edges=y_edges,
            local_grad_x=local_grad_x,
            data_frac=data_frac,
            norm_frac=norm_frac,
            frac_diff=frac_diff,
            data_mean_score_x=data_mean_score_x,
            norm_mean_score_x=norm_mean_score_x,
            score_diff=score_diff,
        )

        print(f"[DALITZ SCORE] Saved arrays: {out_npz}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

            plt.figure(figsize=(8, 7))
            plt.imshow(
                local_grad_x.T,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(label="local grad_x")
            plt.xlabel(r"$m^2(K_S\pi^+)$")
            plt.ylabel(r"$m^2(K_S\pi^-)$")
            plt.title(r"Local contribution to $d\mathrm{NLL}/dx$ at $x=y=0$")
            plt.tight_layout()
            plt.savefig(output_dir / "dalitz_local_grad_x.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 7))
            plt.imshow(
                score_diff.T,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(label="data score_x - norm score_x")
            plt.xlabel(r"$m^2(K_S\pi^+)$")
            plt.ylabel(r"$m^2(K_S\pi^-)$")
            plt.title(r"Difference in mean score_x")
            plt.tight_layout()
            plt.savefig(output_dir / "dalitz_score_x_diff.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 7))
            plt.imshow(
                frac_diff.T,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
            plt.colorbar(label="data fraction - norm fraction")
            plt.xlabel(r"$m^2(K_S\pi^+)$")
            plt.ylabel(r"$m^2(K_S\pi^-)$")
            plt.title("Dalitz density difference")
            plt.tight_layout()
            plt.savefig(output_dir / "dalitz_density_frac_diff.png", dpi=160)
            plt.close()

            print(f"[DALITZ SCORE] Saved plots in: {output_dir}")

        except Exception as exc:
            print(f"[DALITZ SCORE] Could not save plots: {exc}")

    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "local_grad_x": local_grad_x,
        "data_frac": data_frac,
        "norm_frac": norm_frac,
        "frac_diff": frac_diff,
        "data_mean_score_x": data_mean_score_x,
        "norm_mean_score_x": norm_mean_score_x,
        "score_diff": score_diff,
        "total_grad_x": total_grad_x,
    }



def run_score_check_at_zero(
    data_cache,
    norm_cache,
    eps=1.0e-4,
    chunk_size=None,
):
    """Finite-difference score check at x=0,y=0.

    For the normalized likelihood,

        dNLL/dx per event = - <d log pdf / dx>_data + d log(norm_int)/dx

    If this derivative is positive at x=0, the NLL decreases when x moves negative.
    """
    if chunk_size is None:
        chunk_size = _mixing_nll_chunk()


    print()
    print("=" * 100)
    print("DEBUG: score check at x=0, y=0")
    print("=" * 100)
    print(f"[SCORE] finite-difference eps = {eps}")

    n_data_total = int(data_cache["n_events"])
    n_norm_total = int(norm_cache["n_events"])

    # ------------------------------------------------------------
    # Data scores
    # ------------------------------------------------------------
    data_score_x_sum = 0.0
    data_score_y_sum = 0.0
    data_score_x2_sum = 0.0
    data_score_y2_sum = 0.0

    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        A = tf.constant(data_cache["A"][start:stop])
        Abar = tf.constant(data_cache["Abar"][start:stop])
        t = atfi.const(data_cache["t"][start:stop])

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(+eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(-eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)) / atfi.const(2.0 * eps)
        score_y = (tf.math.log(pdf_y_plus) - tf.math.log(pdf_y_minus)) / atfi.const(2.0 * eps)

        score_x_np = score_x.numpy()
        score_y_np = score_y.numpy()

        data_score_x_sum += float(np.sum(score_x_np))
        data_score_y_sum += float(np.sum(score_y_np))
        data_score_x2_sum += float(np.sum(score_x_np * score_x_np))
        data_score_y2_sum += float(np.sum(score_y_np * score_y_np))

    data_mean_score_x = data_score_x_sum / float(n_data_total)
    data_mean_score_y = data_score_y_sum / float(n_data_total)

    data_std_score_x = np.sqrt(data_score_x2_sum / float(n_data_total) - data_mean_score_x**2)
    data_std_score_y = np.sqrt(data_score_y2_sum / float(n_data_total) - data_mean_score_y**2)

    # ------------------------------------------------------------
    # Normalization derivative
    # ------------------------------------------------------------
    norm_sum_0 = 0.0
    norm_sum_x_plus = 0.0
    norm_sum_x_minus = 0.0
    norm_sum_y_plus = 0.0
    norm_sum_y_minus = 0.0

    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        A = tf.constant(norm_cache["A"][start:stop])
        Abar = tf.constant(norm_cache["Abar"][start:stop])
        t = atfi.const(norm_cache["t"][start:stop])

        pdf_0 = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_plus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(+eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_y_minus = mixing_density_cached_amplitudes(
            A, Abar, t,
            atfi.const(0.0),
            atfi.const(-eps),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        norm_sum_0 += float(tf.reduce_sum(pdf_0).numpy())
        norm_sum_x_plus += float(tf.reduce_sum(pdf_x_plus).numpy())
        norm_sum_x_minus += float(tf.reduce_sum(pdf_x_minus).numpy())
        norm_sum_y_plus += float(tf.reduce_sum(pdf_y_plus).numpy())
        norm_sum_y_minus += float(tf.reduce_sum(pdf_y_minus).numpy())

    dlogI_dx = ((norm_sum_x_plus - norm_sum_x_minus) / (2.0 * eps)) / norm_sum_0
    dlogI_dy = ((norm_sum_y_plus - norm_sum_y_minus) / (2.0 * eps)) / norm_sum_0

    grad_x_per_event = -data_mean_score_x + dlogI_dx
    grad_y_per_event = -data_mean_score_y + dlogI_dy

    print()
    print("=" * 100)
    print("Score check summary")
    print("=" * 100)
    print(f"data <d log pdf / dx> = {data_mean_score_x:.10e}")
    print(f"norm d log I / dx     = {dlogI_dx:.10e}")
    print(f"dNLL/dx per event     = {grad_x_per_event:.10e}")
    print(f"dNLL/dx total         = {grad_x_per_event * float(n_data_total):.10e}")
    print()
    print(f"data <d log pdf / dy> = {data_mean_score_y:.10e}")
    print(f"norm d log I / dy     = {dlogI_dy:.10e}")
    print(f"dNLL/dy per event     = {grad_y_per_event:.10e}")
    print(f"dNLL/dy total         = {grad_y_per_event * float(n_data_total):.10e}")
    print()
    print(f"std score x, data     = {data_std_score_x:.10e}")
    print(f"std score y, data     = {data_std_score_y:.10e}")
    print("=" * 100)

    print()
    print("[SCORE] Interpretation:")
    print("[SCORE] If dNLL/dx > 0 at x=0, the NLL decreases by moving x negative.")
    print("[SCORE] If dNLL/dx < 0 at x=0, the NLL decreases by moving x positive.")

    return {
        "data_mean_score_x": data_mean_score_x,
        "data_mean_score_y": data_mean_score_y,
        "dlogI_dx": dlogI_dx,
        "dlogI_dy": dlogI_dy,
        "grad_x_per_event": grad_x_per_event,
        "grad_y_per_event": grad_y_per_event,
        "grad_x_total": grad_x_per_event * float(n_data_total),
        "grad_y_total": grad_y_per_event * float(n_data_total),
        "data_std_score_x": data_std_score_x,
        "data_std_score_y": data_std_score_y,
    }

def run_score_profile_in_time_at_zero(
    data_cache,
    norm_cache,
    eps=1.0e-4,
    n_t_bins=12,
    chunk_size=None,
):
    """Compare the x-score profile in time for data and normalized expectation.

    Data:
        unweighted mean score_x in each t bin.

    Norm:
        weighted mean score_x in each t bin using pdf(x=0,y=0) as weight.
        This represents the expectation from the fitted PDF at zero mixing.
    """
    if chunk_size is None:
        chunk_size = _mixing_nll_chunk()


    print()
    print("=" * 100)
    print("DEBUG: score_x profile in time at x=0, y=0")
    print("=" * 100)

    data_t_min = float(np.min(data_cache["t"]))
    data_t_max = float(np.max(data_cache["t"]))

    t_edges = np.linspace(data_t_min, data_t_max, n_t_bins + 1)

    print(f"[SCORE PROFILE] t range = [{data_t_min:.6f}, {data_t_max:.6f}]")
    print(f"[SCORE PROFILE] n_t_bins = {n_t_bins}")

    data_count = np.zeros(n_t_bins, dtype=np.float64)
    data_score_x_sum = np.zeros(n_t_bins, dtype=np.float64)

    norm_weight_sum = np.zeros(n_t_bins, dtype=np.float64)
    norm_weighted_score_x_sum = np.zeros(n_t_bins, dtype=np.float64)

    # ------------------------------------------------------------
    # Data score profile
    # ------------------------------------------------------------
    n_data_total = int(data_cache["n_events"])

    for start in range(0, n_data_total, chunk_size):
        stop = min(start + chunk_size, n_data_total)

        A = tf.constant(data_cache["A"][start:stop])
        Abar = tf.constant(data_cache["Abar"][start:stop])
        t_tf = atfi.const(data_cache["t"][start:stop])
        t_np = data_cache["t"][start:stop]

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        score_x_np = score_x.numpy()

        bin_idx = np.searchsorted(t_edges, t_np, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < n_t_bins)

        np.add.at(data_count, bin_idx[valid], 1.0)
        np.add.at(data_score_x_sum, bin_idx[valid], score_x_np[valid])

    # ------------------------------------------------------------
    # Norm expected score profile
    # Weighted by pdf(x=0,y=0)
    # ------------------------------------------------------------
    n_norm_total = int(norm_cache["n_events"])

    for start in range(0, n_norm_total, chunk_size):
        stop = min(start + chunk_size, n_norm_total)

        A = tf.constant(norm_cache["A"][start:stop])
        Abar = tf.constant(norm_cache["Abar"][start:stop])
        t_tf = atfi.const(norm_cache["t"][start:stop])
        t_np = norm_cache["t"][start:stop]

        pdf_0 = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(0.0),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_plus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(+eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        pdf_x_minus = mixing_density_cached_amplitudes(
            A, Abar, t_tf,
            atfi.const(-eps),
            atfi.const(0.0),
            atfi.const(1.0),
            atfi.const(0.0),
        )

        score_x = (
            tf.math.log(pdf_x_plus) - tf.math.log(pdf_x_minus)
        ) / atfi.const(2.0 * eps)

        pdf_0_np = pdf_0.numpy()
        score_x_np = score_x.numpy()

        bin_idx = np.searchsorted(t_edges, t_np, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < n_t_bins)

        np.add.at(norm_weight_sum, bin_idx[valid], pdf_0_np[valid])
        np.add.at(
            norm_weighted_score_x_sum,
            bin_idx[valid],
            pdf_0_np[valid] * score_x_np[valid],
        )

    data_mean_score_x = data_score_x_sum / np.maximum(data_count, 1.0)
    norm_mean_score_x = norm_weighted_score_x_sum / np.maximum(norm_weight_sum, 1.0)

    data_frac = data_count / np.sum(data_count)
    norm_frac = norm_weight_sum / np.sum(norm_weight_sum)

    print()
    print("=" * 130)
    print("score_x profile in time")
    print("=" * 130)
    print(
        f"{'bin':>4} {'t_low':>10} {'t_high':>10} "
        f"{'data_frac':>12} {'norm_frac':>12} "
        f"{'data_score_x':>16} {'norm_score_x':>16} {'diff':>16}"
    )
    print("-" * 130)

    for i in range(n_t_bins):
        diff = data_mean_score_x[i] - norm_mean_score_x[i]

        print(
            f"{i:>4d} "
            f"{t_edges[i]:>10.5f} "
            f"{t_edges[i+1]:>10.5f} "
            f"{data_frac[i]:>12.6e} "
            f"{norm_frac[i]:>12.6e} "
            f"{data_mean_score_x[i]:>16.8e} "
            f"{norm_mean_score_x[i]:>16.8e} "
            f"{diff:>16.8e}"
        )

    print("=" * 130)

    return {
        "t_edges": t_edges,
        "data_frac": data_frac,
        "norm_frac": norm_frac,
        "data_mean_score_x": data_mean_score_x,
        "norm_mean_score_x": norm_mean_score_x,
    }




DIAGNOSTIC_FUNCTION_NAMES = [
    "make_fixed_mixing_pars",
    "evaluate_cached_nll_terms",
    "finite_difference_x_diagnostics",
    "run_fixed_point_nll_check",
    "run_x_scan_nll_check",
    "run_score_profile_in_dalitz_at_zero",
    "run_score_check_at_zero",
    "run_score_profile_in_time_at_zero",
]
