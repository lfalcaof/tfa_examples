import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gengamma
from scipy.integrate import simpson

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
import amplitf.interface as atfi


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../output/plots/acceptance_studies"))
os.makedirs(PLOT_DIR, exist_ok=True)


try:
    import lp
    mkz = atfi.const(lp.K_S_0.mass/1000.0)
    mpi = atfi.const(lp.pi_plus.mass/1000.0)
    md  = atfi.const(lp.D_0.mass/1000.0)
except Exception:
    mkz = atfi.const(0.497611)
    mpi = atfi.const(0.13957039)
    md  = atfi.const(1.86483)

@tf.function
def generalized_gamma(shape, d, p, a, dtype=tf.float32, seed=None):
    """
    Samples from the generalized gamma distribution.
    
    Parameters:
    - shape: output tensor shape
    - d: shape parameter (float or tensor)
    - p: power parameter (float or tensor)
    - a: scale parameter (float or tensor)
    - dtype: tf.float32 or tf.float64
    - seed: optional random seed
    
    Returns:
    - samples: Tensor of shape `shape`, dtype `dtype`
    """
    d = tf.convert_to_tensor(d, dtype=dtype)
    p = tf.convert_to_tensor(p, dtype=dtype)
    a = tf.convert_to_tensor(a, dtype=dtype)

    gamma_dist = tfd.Gamma(concentration=d, rate=1.0)
    u = gamma_dist.sample(shape, seed=seed)
    
    # Transform: X = a * U^(1/p)  
    x = a * tf.pow(u, 1.0 / p)
    return x

# -----------------------------------------------------------------------------
# Target acceptance (for fitting reference)
# -----------------------------------------------------------------------------
def target_pdf(u, a, b, n, m, beta):
    return (u**a / (b + u**n))**m * np.exp(-beta * u)

def fit_gengamma_to_acceptance():
    """
    Fit a generalized gamma to a target acceptance-only PDF in u = t - t0.
    Returns (shape_c, shape_a, loc, scale, t0).
    """
    # Parameters
    a = 2
    b = 1
    n = 3
    m = 1.5
    beta = 1.0
    t0 = 0.5

    # 1) Grid in u = t - t0
    u_grid = np.linspace(1e-5, 10, 1000)
    pdf_vals = target_pdf(u_grid, a, b, n, m, beta)

    # 2) Numerical normalization
    Z = simpson(pdf_vals, x=u_grid)
    pdf_vals /= Z

    # 3) Approximate sampling via numeric inverse CDF
    rng = np.random.default_rng(42)
    cdf = np.cumsum(pdf_vals); cdf /= cdf[-1]
    inverse_cdf = lambda y: np.interp(y, cdf, u_grid)
    samples_from_pdf = np.array([inverse_cdf(y) for y in rng.uniform(0, 1, 10000)])

    # 4) Fit generalized gamma
    shape_a, shape_c, loc, scale = gengamma.fit(samples_from_pdf, floc=0)

    print("Fitted generalized gamma parameters:")
    print(f"  shape c:     {shape_c}")
    print(f"  shape a:     {shape_a}")
    print(f"  loc:         {loc}")
    print(f"  scale:       {scale}")

    # 5) Quick comparison figure
    samples = gengamma.rvs(c=shape_c, a=shape_a, loc=loc, scale=scale, size=10000)
    t_samples = t0 + samples

    plt.figure(figsize=(8, 5))
    plt.hist(t_samples - t0, bins=100, density=True, alpha=0.5, label="Fitted gengamma samples")
    plt.plot(u_grid, pdf_vals, label="Target PDF (normalized)")
    plt.xlabel("u = t - t0"); plt.ylabel("Density")
    plt.legend(); plt.title("Comparison: Fitted Generalized Gamma vs Target PDF")
    plt.grid(True)
    out_path = os.path.join(PLOT_DIR, "gengamma_fit_vs_target.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Figure saved to: {out_path}")

    return shape_c, shape_a, loc, scale, t0

# -----------------------------------------------------------------------------
# Acceptance visualization (decay time)
# -----------------------------------------------------------------------------
def dt_acceptance(t, t0, a, b, beta, m, n):
    dt = t - t0
    return np.where(dt >= 0, (dt**a / (b + dt**n))**m * np.exp(-beta * dt), 0)

def plot_dt_acceptance():
    t0 = 0.335
    a, b, beta, m, n = 1.14, 6.73, 1.27, 2.97, 0.17
    t = np.linspace(0, 10, 1000)
    y = dt_acceptance(t, t0, a, b, beta, m, n)

    plt.figure(figsize=(8,5))
    plt.plot(t, y)
    plt.xlabel("t"); plt.ylabel("Acceptance")
    plt.yscale("log"); plt.title("Decay Time Acceptance Function")
    plt.grid(True)
    out_path = os.path.join(PLOT_DIR, "dt_acceptance_function.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Figure saved to: {out_path}")

# -----------------------------------------------------------------------------
# Dalitz helpers (uniform base sampling)
# -----------------------------------------------------------------------------
def sample_dalitz(size):
    """
    Return DalitzPhaseSpace object and a uniform-in-box filtered sample s ~ DP.
    s has shape [N, 2] with columns [m2ab, m2bc], N ≤ size.
    """
    dp = DalitzPhaseSpace(mpi, mkz, mpi, md)
    return dp, dp.uniform_sample(size)

@tf.function
def normalize_zz(m2ac, minac, maxac):
    """
    Normalize m2ac into [0,1] and clip.
    """
    z = (m2ac - minac) / (maxac - minac)
    return tf.clip_by_value(z, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Dalitz efficiency favoring low m2(pi pi)
#   z = normalized m2(pi pi) in [0,1]
#   eps(z) = max(eps_min, (1 - z)**alpha)  -> high near z=0, low near z=1
# -----------------------------------------------------------------------------
@tf.function
def dalitz_efficiency_prob(dp, s, alpha=1.0, eps_min=0.0):
    """
    dp : DalitzPhaseSpace
    s  : [N,2] TF tensor (m2ab, m2bc)
    alpha   : controls how fast efficiency falls with m2(pi pi); alpha>0
    eps_min : floor probability to avoid zero-acceptance tails
    Returns per-event acceptance probabilities in [eps_min, 1].
    """
    m2ac = dp.m2ac(s)  # [N]
    z = normalize_zz(m2ac, dp.minac, dp.maxac)   # 0 at low m2pipi, 1 at high m2pipi
    eps = tf.pow(1.0 - z, tf.cast(alpha, z.dtype))
    if eps_min > 0.0:
        eps = tf.maximum(eps, tf.cast(eps_min, eps.dtype))
    return eps, m2ac

@tf.function
def apply_dalitz_efficiency(dp, s, alpha=1.0, eps_min=0.0):
    """
    Accept-reject filtering of Dalitz points with prob eps(s).
    Returns filtered s, m2ac, eps.
    """
    eps, m2ac = dalitz_efficiency_prob(dp, s, alpha=alpha, eps_min=eps_min)
    r = tf.random.uniform(tf.shape(eps), dtype=eps.dtype)
    mask = r < eps
    return tf.boolean_mask(s, mask), tf.boolean_mask(m2ac, mask), tf.boolean_mask(eps, mask)

def sample_dalitz_with_eff(size, alpha=1.0, eps_min=0.0, oversample_factor=2.0):
    """
    Generate ~'size' accepted Dalitz points by oversampling then accept-reject.
    Increase oversample_factor if you want the final sample closer to 'size'.
    """
    dp = DalitzPhaseSpace(mpi, mkz, mpi, md)
    attempt = int(max(size, int(size * oversample_factor)))
    s0 = dp.uniform_sample(attempt)                    
    s, m2ac, _ = apply_dalitz_efficiency(dp, s0, alpha=alpha, eps_min=eps_min)
    if s.shape[0] < size:
        extra = int(size - int(s.shape[0]))
        s1 = dp.uniform_sample(max(extra, attempt // 2))
        s_add, m2ac_add, _ = apply_dalitz_efficiency(dp, s1, alpha=alpha, eps_min=eps_min)
        s   = tf.concat([s, s_add], axis=0)
        m2ac = tf.concat([m2ac, m2ac_add], axis=0)
    s = s[:size]
    m2ac = m2ac[:size]
    return dp, s, m2ac

# -----------------------------------------------------------------------------
# UNIFIED MODEL: sampler(s)->t and joint generator (s, t)
# -----------------------------------------------------------------------------
class TimeModel:
    """
    Unified model that:
      - provides sample_t_for_s(s)  -> t [N,1] (for toymc.py)
      - provides generate(size, ...)-> (dp, s, m2ac, t, joint) (for tests/plots)
    The correlation is controlled by 'rho' via the per-event scale
    a_vec = a_base * (1 + rho * z'), with z' = 1 - normalize(m2ac).
    """
    def __init__(self, rho=0.8):
        shape_c, shape_a, loc, scale, t0 = fit_gengamma_to_acceptance()
        self.d, self.p, self.a_base = shape_a, shape_c, scale
        self.t0 = t0
        self.rho = rho
        self.fp = atfi.fptype()

        self.tphsp = DecayTimePhaseSpace(atfi.const(1.0))
        self.dp    = DalitzPhaseSpace(mpi, mkz, mpi, md)

    @tf.function
    def _a_vec_for_s(self, s):
        # s: [N,2] -> m2ac -> z' in [0,1] -> per-event a_vec
        m2ac = self.dp.m2ac(s)  # [N]
        z = 1.0 - normalize_zz(m2ac, self.dp.minac, self.dp.maxac)  # z' = 1 - z
        a_vec = tf.cast(self.a_base, self.fp) * (1.0 + tf.cast(self.rho, self.fp) * tf.cast(z, self.fp))
        return a_vec, m2ac

    @tf.function
    def sample_t_for_s(self, s):
        """s: [N,2] -> returns t: [N,1]"""
        a_vec, _ = self._a_vec_for_s(s)
        pars = (tf.cast(self.d, self.fp), tf.cast(self.p, self.fp), a_vec)
        u = self.tphsp.acceptance_sample(generalized_gamma, pars, tf.shape(a_vec)[0])  # [N,1]
        t = tf.cast(self.t0, self.fp) + tf.reshape(u, [-1])
        return tf.reshape(t, [-1, 1])  # [N,1]

    def generate(self, size=200_000, use_dalitz_eff=True, alpha=1.4, eps_min=0.05, oversample_factor=2.0):
        """
        Generate a joint sample as in the previous generate_correlated_samples.
        If use_dalitz_eff=True, apply Dalitz-space efficiency depending on m2(pi pi)
        with eps(z) = max(eps_min, (1 - z)^alpha).
        Returns: dp, s, m2ac, t, joint ([N,3] = m2ab, m2bc, t).
        """
        if use_dalitz_eff:
            dp, s, m2ac = sample_dalitz_with_eff(size, alpha=alpha, eps_min=eps_min,
                                                 oversample_factor=oversample_factor)
            self.dp = dp
        else:
            dp, s = sample_dalitz(size)
            self.dp = dp
            # m2ac for consistent return
            m2ac = self.dp.m2ac(s)

        a_vec, m2ac = self._a_vec_for_s(s)       
        pars = (tf.cast(self.d, self.fp), tf.cast(self.p, self.fp), a_vec)
        u = self.tphsp.acceptance_sample(generalized_gamma, pars, tf.shape(a_vec)[0])  
        t = tf.cast(self.t0, self.fp) + tf.reshape(u, [-1])                           
        joint = tf.concat([s, tf.reshape(t, [-1,1])], axis=1)                          
        return self.dp, s, m2ac, t, joint

def make_time_model(rho=0.8):
    """Single entry point replacing make_time_sampler and generate_correlated_samples."""
    return TimeModel(rho=rho)

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_dalitz_scatter(s, fname):
    pts = s.numpy()
    plt.figure(figsize=(6,5))
    plt.scatter(pts[:,0], pts[:,1], s=2, alpha=0.35)
    plt.xlabel(r"$m^2(K_S\pi)_1$"); plt.ylabel(r"$m^2(K_S\pi)_2$")
    plt.title("Dalitz phase space")
    out = os.path.join(PLOT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Figure saved to: {out}")

def plot_dalitz_colored_by_t(dp, s, t, fname):
    pts = s.numpy(); tt = t.numpy()
    plt.figure(figsize=(6,5))
    sc = plt.scatter(pts[:,0], pts[:,1], c=tt, s=2, alpha=0.6)
    plt.xlabel(r"$m^2(K_S\pi)_+$"); plt.ylabel(r"$m^2(K_S\pi)_-$")
    plt.title("Dalitz colored by decay time t")
    cbar = plt.colorbar(sc); cbar.set_label("t")
    out = os.path.join(PLOT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Figure saved to: {out}")

def plot_t_vs_pipi(dp, m2ac, t, fname):
    x = m2ac.numpy(); y = t.numpy()
    nb = 40
    bins = np.linspace(dp.minac, dp.maxac, nb+1)
    idx = np.digitize(x, bins) - 1
    means = np.array([y[idx==i].mean() if np.any(idx==i) else np.nan for i in range(nb)])
    centers = 0.5*(bins[:-1]+bins[1:])
    plt.figure(figsize=(7,4))
    plt.plot(centers, means, marker="o", lw=1)
    plt.xlabel(r"$m^2(\pi\pi)$"); plt.ylabel(r"$\mathbb{E}[t\,|\,m^2(\pi\pi)]$")
    plt.title("Mean decay time vs $m^2(\\pi\\pi)$"); plt.grid(True)
    out = os.path.join(PLOT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Figure saved to: {out}")


def main():
    # Analytical plot for acceptance in t
    plot_dt_acceptance()

    # Unified model
    model = make_time_model(rho=0.8)

    # 1) Generate joint sample with Dalitz-space efficiency (denser at low m2(pi pi))
    dp, s, m2ac, t, joint = model.generate(
        size=100_000,
        use_dalitz_eff=True,   # toggle Dalitz-space efficiency on/off
        alpha=2.0,             # >1 falls faster with m2(pi pi)
        eps_min=0.05,          # floor to avoid zero acceptance at high m2(pi pi)
        oversample_factor=2.0  # increases chance to hit the target size
    )


    # Figures
    plot_dalitz_scatter(s, "dalitz_phsp_scatter_correlated_input.png")
    plot_dalitz_colored_by_t(dp, s, t, "dalitz_colored_by_t.png")
    plot_t_vs_pipi(dp, m2ac, t, "mean_t_vs_m2pipi.png")

    # Save the joint sample
    out_npz = os.path.join(PLOT_DIR, "joint_sample_m2ab_m2bc_t.npz")
    np.savez_compressed(out_npz,
                        m2ab=s.numpy()[:,0],
                        m2bc=s.numpy()[:,1],
                        t=t.numpy())
    print(f"Joint sample saved to: {out_npz}")

if __name__ == "__main__":
    main()
