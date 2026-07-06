from __future__ import annotations

from typing import Any

from d02kspipi_fitcore.runtime.context import FitContext


def build_complex_amplitude_model_and_projection(context: FitContext) -> None:
    """Define the complex Dalitz amplitude model and optional projection plots.

    This is the native implementation of the validated v10/v11
    ``step_08_complex_projection`` block.  It intentionally keeps the same
    legacy symbol names in ``FitContext.namespace`` so later preserved and
    native blocks continue to see the same runtime state.
    """

    ns: dict[str, Any] = context.namespace

    # Frequently used runtime symbols.  These are read once here and captured
    # by the nested amplitude_model closure, matching the original global-name
    # behavior while avoiding process-level globals.
    np = ns["np"]
    atfi = ns["atfi"]
    atfd = ns["atfd"]

    phsp = ns["phsp"]
    helicity_babar2008 = ns["helicity_babar2008"]
    LASS_babar2008 = ns["LASS_babar2008"]

    fit_switches = ns["fit_switches"]

    # Resonance constants and line-shape parameters.
    mrho = ns["mrho"]
    wrho = ns["wrho"]
    mkst = ns["mkst"]
    wkst = ns["wkst"]
    mk2st1430 = ns["mk2st1430"]
    wk2st1430 = ns["wk2st1430"]
    mkst1410 = ns["mkst1410"]
    wkst1410 = ns["wkst1410"]
    mkst1680 = ns["mkst1680"]
    wkst1680 = ns["wkst1680"]
    momega = ns["momega"]
    womega = ns["womega"]
    mf2_1270 = ns["mf2_1270"]
    wf2_1270 = ns["wf2_1270"]
    mrho1450 = ns["mrho1450"]
    wrho1450 = ns["wrho1450"]

    lass_a = ns["lass_a"]
    lass_r = ns["lass_r"]
    lass_M = ns["lass_M"]
    lass_G = ns["lass_G"]
    lass_R = ns["lass_R"]
    lass_phiR = ns["lass_phiR"]
    lass_F = ns["lass_F"]
    lass_phiF = ns["lass_phiF"]

    m_poles = ns["m_poles"]
    g_poles = ns["g_poles"]
    s0 = ns["s0"]
    fij = ns["fij"]
    b_poles = ns["b_poles"]
    K_matrix_sprod = ns["K_matrix_sprod"]
    fprod1 = ns["fprod1"]
    mpi = ns["mpi"]
    mkz = ns["mkz"]
    meta = ns["meta"]
    metap = ns["metap"]

    def amplitude_model(x):
        """Complex amplitude A(m2ab, m2bc) using the validated BABAR model."""

        m2ab = phsp.m2ab(x)
        m2bc = phsp.m2bc(x)
        m2ac = phsp.m2ac(x)

        mab = atfi.sqrt(m2ab)
        mbc = atfi.sqrt(m2bc)
        mac = atfi.sqrt(m2ac)

        # BABAR 2008 helicity convention.
        hel_ab_1 = -atfi.cast_complex(
            helicity_babar2008(mab, mbc, mac, phsp.ma, phsp.mb, phsp.mc, phsp.md, 1)
        )
        hel_bc_1 = -atfi.cast_complex(
            helicity_babar2008(mbc, mab, mac, phsp.mc, phsp.mb, phsp.ma, phsp.md, 1)
        )
        hel_ac_1 = -atfi.cast_complex(
            helicity_babar2008(mac, mbc, mab, phsp.ma, phsp.mc, phsp.mb, phsp.md, 1)
        )

        hel_ab_2 = atfi.cast_complex(
            helicity_babar2008(mab, mbc, mac, phsp.ma, phsp.mb, phsp.mc, phsp.md, 2)
        )
        hel_bc_2 = atfi.cast_complex(
            helicity_babar2008(mbc, mab, mac, phsp.mc, phsp.mb, phsp.ma, phsp.md, 2)
        )
        hel_ac_2 = atfi.cast_complex(
            helicity_babar2008(mac, mbc, mab, phsp.ma, phsp.mc, phsp.mb, phsp.md, 2)
        )

        bw1 = atfd.breit_wigner_lineshape(
            m2ac, mrho, wrho,
            phsp.ma, phsp.mc, phsp.mb, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw2 = atfd.breit_wigner_lineshape(
            m2ab, mkst, wkst,
            phsp.ma, phsp.mb, phsp.mc, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw4 = atfd.breit_wigner_lineshape(
            m2ab, mk2st1430, wk2st1430,
            phsp.ma, phsp.mb, phsp.mc, phsp.md,
            1.5, 5.0, 2, 2,
            barrier_factor=False,
        )

        bw5 = atfd.breit_wigner_lineshape(
            m2ab, mkst1410, wkst1410,
            phsp.ma, phsp.mb, phsp.mc, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw6 = atfd.breit_wigner_lineshape(
            m2ab, mkst1680, wkst1680,
            phsp.ma, phsp.mb, phsp.mc, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw7 = atfd.breit_wigner_lineshape(
            m2bc, mkst, wkst,
            phsp.mc, phsp.mb, phsp.ma, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw9 = atfd.breit_wigner_lineshape(
            m2bc, mk2st1430, wk2st1430,
            phsp.mc, phsp.mb, phsp.ma, phsp.md,
            1.5, 5.0, 2, 2,
            barrier_factor=False,
        )

        bw10 = atfd.breit_wigner_lineshape(
            m2bc, mkst1410, wkst1410,
            phsp.mc, phsp.mb, phsp.ma, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw11 = atfd.breit_wigner_lineshape(
            m2ac, momega, womega,
            phsp.ma, phsp.mc, phsp.mb, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
        )

        bw12 = atfd.breit_wigner_lineshape(
            m2ac, mf2_1270, wf2_1270,
            phsp.ma, phsp.mc, phsp.mb, phsp.md,
            1.5, 5.0, 2, 2,
            barrier_factor=False,
        )

        bw13 = atfd.breit_wigner_lineshape(
            m2ac, mrho1450, wrho1450,
            phsp.ma, phsp.mc, phsp.mb, phsp.md,
            1.5, 5.0, 1, 1,
            barrier_factor=False,
            md0=mrho1450 + phsp.mb,
        )

        # LASS.
        lass_n = LASS_babar2008(
            m2ab, lass_a, lass_r, lass_M, lass_G,
            phsp.mb, phsp.ma,
            lass_R, lass_phiR, lass_F, lass_phiF,
        )

        lass_p = LASS_babar2008(
            m2bc, lass_a, lass_r, lass_M, lass_G,
            phsp.mb, phsp.mc,
            lass_R, lass_phiR, lass_F, lass_phiF,
        )

        # K matrix.
        km = atfd.kmatrix_lineshape(
            m2ac,
            m_poles,
            g_poles,
            s0,
            fij,
            b_poles,
            K_matrix_sprod,
            fprod1,
            [[mpi, mpi], [mkz, mkz], [mpi], [meta, meta], [meta, metap]],
        )

        def _model(
            a1r,
            a1i,
            a2r,
            a2i,
            a3r,
            a3i,
            a4r,
            a4i,
            a5r,
            a5i,
            a6r,
            a6i,
            a7r,
            a7i,
            a8r,
            a8i,
            a9r,
            a9i,
            a10r,
            a10i,
            a11r,
            a11i,
            a12r,
            a12i,
            a13r,
            a13i,
            a14r,
            a14i,
            switches=fit_switches,
        ):
            a1 = atfi.complex(a1r, a1i)
            a2 = atfi.complex(a2r, a2i)
            a3 = atfi.complex(a3r, a3i)
            a4 = atfi.complex(a4r, a4i)
            a5 = atfi.complex(a5r, a5i)
            a6 = atfi.complex(a6r, a6i)
            a7 = atfi.complex(a7r, a7i)
            a8 = atfi.complex(a8r, a8i)
            a9 = atfi.complex(a9r, a9i)
            a10 = atfi.complex(a10r, a10i)
            a11 = atfi.complex(a11r, a11i)
            a12 = atfi.complex(a12r, a12i)
            a13 = atfi.complex(a13r, a13i)
            a14 = atfi.complex(a14r, a14i)

            ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(0.0), atfi.const(0.0)
            )

            if switches[0]:
                ampl += a1 * bw1 * hel_ac_1
            if switches[1]:
                ampl += a2 * bw2 * hel_ab_1
            if switches[2]:
                ampl += a3 * lass_n
            if switches[3]:
                ampl += a4 * bw4 * hel_ab_2
            if switches[4]:
                ampl += a5 * bw5 * hel_ab_1
            if switches[5]:
                ampl += a6 * bw6 * hel_ab_1
            if switches[6]:
                ampl += a7 * bw7 * hel_bc_1
            if switches[7]:
                ampl += a8 * lass_p
            if switches[8]:
                ampl += a9 * bw9 * hel_bc_2
            if switches[9]:
                ampl += a10 * bw10 * hel_bc_1
            if switches[10]:
                ampl += a11 * bw11 * hel_ac_1
            if switches[11]:
                ampl += a12 * bw12 * hel_ac_2
            if switches[12]:
                ampl += a13 * bw13 * hel_ac_1
            if switches[13]:
                ampl += a14 * km
            if switches[14]:
                ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                    atfi.const(5.0), atfi.const(0.0)
                )

            return ampl

        return _model

    ns["amplitude_model"] = amplitude_model

    if ns.get("CREATE_FITTED_SAMPLE"):
        _build_fitted_sample_projection(ns, amplitude_model)


def _build_fitted_sample_projection(ns: dict[str, Any], amplitude_model) -> None:
    """Native fitted-sample projection block from v10/v11 step_08."""

    np = ns["np"]
    atfi = ns["atfi"]
    tfp = ns["tfp"]
    phsp = ns["phsp"]

    print_section = ns["print_section"]
    save_figure = ns["save_figure"]

    COMPONENTS = ns["COMPONENTS"]
    SWITCH = ns["SWITCH"]
    nnorm_dalitz_effective = ns["nnorm_dalitz_effective"]
    integration_sample = ns["integration_sample"]
    fitted_model = ns["fitted_model"]
    ARRAY_DIR = ns["ARRAY_DIR"]
    data_tf_small = ns["data_tf_small"]

    print_section("Build fitted sample for projection plots")

    amps_to_plot = [
        comp["id"] - 1
        for comp in COMPONENTS
        if SWITCH[comp["key"]]
    ]

    if SWITCH["const"]:
        amps_to_plot.append(14)

    n_component_columns = max(amps_to_plot) + 1
    n_toy = max(1, int(nnorm_dalitz_effective / 100))

    integration_np_for_plot = (
        integration_sample.numpy() if hasattr(integration_sample, "numpy") else integration_sample
    )
    integration_np_for_plot = np.asarray(integration_np_for_plot)[:, :2]

    eval_chunk = 500000
    weights_list = []

    for i in range(0, len(integration_np_for_plot), eval_chunk):
        x_chunk = atfi.const(integration_np_for_plot[i : i + eval_chunk])
        w_chunk = fitted_model(x_chunk).numpy()
        w_chunk = np.real(w_chunk)
        w_chunk[w_chunk < 0.0] = 0.0
        weights_list.append(w_chunk)

    weights = np.concatenate(weights_list)
    weights_sum = np.sum(weights)
    if weights_sum <= 0.0:
        raise RuntimeError("Fitted-model weights have non-positive sum.")
    weights = weights / weights_sum

    idx = np.random.choice(
        np.arange(len(integration_np_for_plot)),
        size=n_toy,
        replace=True,
        p=weights,
    )

    toy_xy = integration_np_for_plot[idx]

    total = fitted_model(atfi.const(toy_xy)).numpy()
    total = np.real(total)
    total[total <= 0.0] = 1e-12

    component_columns = np.zeros((len(toy_xy), n_component_columns))

    for amp_index in amps_to_plot:
        component_switches = [False] * 15
        component_switches[amp_index] = True

        comp = fitted_model(
            atfi.const(toy_xy),
            switches=component_switches,
        ).numpy()

        comp = np.real(comp)
        comp[comp < 0.0] = 0.0
        component_columns[:, amp_index] = comp / total

    fitted_sample = np.column_stack([
        toy_xy,
        component_columns,
    ])

    print("fitted_sample shape:", fitted_sample.shape)
    np.save(ARRAY_DIR / "fitted_sample_projection.npy", fitted_sample)

    def make_fit_comparison_plot():
        import matplotlib.pyplot as plt

        tfp.set_lhcb_style(size=12, usetex=False)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

        amps_to_plot_local = list(amps_to_plot)

        tfp.plot_distr2d(
            data_tf_small[:, 1],
            data_tf_small[:, 0],
            bins=(50, 50),
            ranges=((0.3, 3.1), (0.3, 3.1)),
            fig=fig,
            ax=ax[0, 0],
            labels=(r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^-)$"),
            units=("MeV$^2$", "MeV$^2$"),
            log=True,
        )

        tfp.plot_distr1d_comparison(
            data_tf_small[:, 1],
            fitted_sample[:, 1],
            cweights=[fitted_sample[:, 2 + i] for i in amps_to_plot_local],
            bins=50,
            range=(0.3, 3.1),
            ax=ax[0, 1],
            label=r"$m^2(K_S^0\pi^+)$",
            units="MeV$^2$",
        )

        tfp.plot_distr1d_comparison(
            data_tf_small[:, 0],
            fitted_sample[:, 0],
            cweights=[fitted_sample[:, 2 + i] for i in amps_to_plot_local],
            bins=50,
            range=(0.3, 3.1),
            ax=ax[1, 0],
            label=r"$m^2(K_S^0\pi^-)$",
            units="MeV$^2$",
        )

        tfp.plot_distr1d_comparison(
            phsp.m2ac(data_tf_small),
            phsp.m2ac(fitted_sample),
            cweights=[fitted_sample[:, 2 + i] for i in amps_to_plot_local],
            bins=50,
            range=(0.05, 1.9),
            ax=ax[1, 1],
            label=r"$m^2(\pi^+\pi^-)$",
            units="MeV$^2$",
            log=True,
        )

        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        return fig

    try:
        fig_fit = make_fit_comparison_plot()
        save_figure(fig_fit, "fit_comparison_dalitz")
        import matplotlib.pyplot as plt

        plt.close(fig_fit)
        print("[OUTPUT] Saved fit comparison figure.")
    except Exception as exc:
        print(f"[WARNING] Could not save fit comparison figure: {exc}")

    ns["amps_to_plot"] = amps_to_plot
    ns["n_component_columns"] = n_component_columns
    ns["n_toy"] = n_toy
    ns["fitted_sample"] = fitted_sample
    ns["make_fit_comparison_plot"] = make_fit_comparison_plot
