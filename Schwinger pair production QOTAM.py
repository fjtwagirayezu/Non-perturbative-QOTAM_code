# ============================================================
# QOTAM Phase-4B: Schwinger Pair Production via QOTAM
# ------------------------------------------------------------
# Goal: Produce the Schwinger/Sauter pair spectrum n_k and integrated yield
#       using a QOTAM optimization, not direct Dirac evolution.
#
# Key idea (QOTAM):
#  - Treat each momentum mode k as a two-outcome channel:
#      "no pair" with prob (1 - n_k), "pair" with prob n_k.
#  - Define an action-derived dynamical cost S(k;E0,tau) (worldline/WKB prior).
#  - Minimize an entropic objective (free energy) per mode:
#      F[n_k] = n_k S(k) + n_k ln n_k + (1-n_k) ln(1-n_k)
#    (up to weights). The minimizer is:
#      n_k = 1 / (1 + exp(S(k)))    (fermionic logistic)
#  - This is non-perturbative (non-analytic in 1/E0) and enforces 0<=n_k<=1.
#
# Background field (Sauter pulse):
#   E(t) = E0 / cosh^2(t/tau)
#   The pulse enters via the Keldysh-like parameter gamma_k.
#
# We also test gauge equivalence:
#   A(t)->A(t)+A0 corresponds to shifting canonical momentum k->k + eA0.
#
# Outputs:
#  - n_k spectrum plot (PDF)
#  - log(n_total) vs 1/E0 plot (PDF)
#  - gauge-equivalence spectrum overlay (PDF)
#  - table CSV of results
#
# Requirements: numpy, matplotlib
# ============================================================

import os, math, datetime
import numpy as np
import matplotlib.pyplot as plt

# ==================== FONT SIZE SETTINGS (Julia-style) ====================
plt.rcParams.update({
    'font.size':         12,   # base font size
    'axes.titlesize':    15,   # titlefontsize
    'axes.labelsize':    14,   # guidefontsize (xlabel, ylabel)
    'xtick.labelsize':   12,   # tickfontsize
    'ytick.labelsize':   12,   # tickfontsize
    'legend.fontsize':   12,   # legendfontsize
    'figure.titlesize':  15,
})
# ========================================================================

# Larger figure size recommended for bigger fonts
FIGSIZE = (9, 5.5)

# -----------------------------
# Helpers: output
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_pdf(fig, outdir, name):
    ensure_dir(outdir)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    print(f"Saved: {outdir}/{name}.pdf")

# -----------------------------
# QOTAM dynamical cost for Schwinger/Sauter
# -----------------------------
def g_sauter(gamma):
    """
    Pulse-shape correction factor (common worldline/WKB form for Sauter-like pulses):
      g(gamma) = 2 / (1 + sqrt(1 + gamma^2))
    Limits:
      gamma -> 0 : g -> 1 (constant-field tunneling limit)
      gamma large: g ~ 2/gamma  (multiphoton-ish regime)
    """
    return 2.0 / (1.0 + np.sqrt(1.0 + gamma**2))

def S_inst_k(k, E0, tau, m=1.0, e=1.0):
    """
    Action-like cost for creating a pair in mode k (1+1D proxy, k is canonical momentum).
    Use p_perp=0; you can generalize by replacing m^2 -> m^2 + k_perp^2.
    We use:
      S(k) = pi * (m^2 + k^2) / (eE0) * g(gamma_k),
      gamma_k = sqrt(m^2 + k^2) / (e E0 tau)
    This yields the correct qualitative non-analytic dependence and pulse curvature.
    """
    omega_k = np.sqrt(m*m + k*k)
    gamma_k = omega_k / (e * E0 * tau)
    return math.pi * (m*m + k*k) / (e * E0) * g_sauter(gamma_k)

def qotam_nk_spectrum(kgrid, E0, tau, m=1.0, e=1.0):
    """
    QOTAM solution for fermionic pair probability per mode:
      n_k = 1/(1 + exp(S(k)))
    Derived by minimizing entropic objective per mode.
    """
    Svals = np.array([S_inst_k(k, E0, tau, m=m, e=e) for k in kgrid], dtype=float)
    # Logistic form ensures 0 < n_k < 1 (unitarity per mode)
    nk = 1.0 / (1.0 + np.exp(Svals))
    return nk, Svals

def integrate_n_total_1d(kgrid, nk):
    """
    1+1D pair density proxy: n_total = ∫ dk/(2pi) n_k
    """
    dk = kgrid[1] - kgrid[0]
    return float(np.sum(nk) * dk / (2.0 * math.pi))

# -----------------------------
# Gauge-equivalence check for the QOTAM spectrum
# -----------------------------
def gauge_equivalence_qotam(kgrid, E0, tau, A0, m=1.0, e=1.0):
    nk0, _ = qotam_nk_spectrum(kgrid, E0, tau, m=m, e=e)
    # shifted gauge: spectrum at same kgrid should equal unshifted at k+eA0
    nk_shifted, _ = qotam_nk_spectrum(kgrid + e*A0, E0, tau, m=m, e=e)
    # Compare integrated totals
    n0 = integrate_n_total_1d(kgrid, nk0)
    n1 = integrate_n_total_1d(kgrid, nk_shifted)
    rel = abs(n1 - n0) / max(abs(n0), 1e-300)
    return nk0, nk_shifted, n0, n1, rel

# ============================================================
# Run QOTAM Schwinger benchmark and save plots
# ============================================================
def run_qotam_schwinger(
    E0_list=(0.12, 0.16, 0.22, 0.30, 0.42, 0.60),
    tau=2.5,
    m=1.0,
    e=1.0,
    Nk=600,
    kmax=7.5,
    A0=0.8
):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ensure_dir(os.path.join("qotam_schwinger_qotam_outputs", f"run_{stamp}"))
    print(f"Output directory:\n  {outdir}")

    kgrid = np.linspace(-kmax, kmax, Nk)

    rows = []
    print("\nComputing QOTAM Schwinger/Sauter spectra...")
    for E0 in E0_list:
        nk, Svals = qotam_nk_spectrum(kgrid, E0, tau, m=m, e=e)
        n_total = integrate_n_total_1d(kgrid, nk)

        # Vacuum persistence proxy assuming independent Bernoulli modes:
        logP0 = float(np.sum(np.log(np.maximum(1.0 - nk, 1e-300))))
        P0 = float(np.exp(logP0))

        # Simple unitarity check: nk in [0,1]
        nk_min, nk_max = float(np.min(nk)), float(np.max(nk))
        rows.append((E0, 1.0/E0, n_total, math.log(max(n_total,1e-300)), P0, nk_min, nk_max))

        print(f"E0={E0: .3f}  n_total={n_total:.6e}  P0={P0:.6e}  nk_range=[{nk_min:.3e},{nk_max:.3e}]")

    # Save CSV table
    csv_path = os.path.join(outdir, "qotam_schwinger_table.csv")
    with open(csv_path, "w") as f:
        f.write("E0,invE0,n_total,log_n_total,P0,nk_min,nk_max\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved: {csv_path}")

    # Plot 1: n_k spectrum for a representative E0 (middle of list)
    E0_mid = E0_list[len(E0_list)//2]
    nk_mid, _ = qotam_nk_spectrum(kgrid, E0_mid, tau, m=m, e=e)

    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(kgrid, nk_mid)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$n_k$ (QOTAM)")
    plt.title(f"QOTAM Schwinger/Sauter spectrum ($E_0$={E0_mid:.3f}, $\\tau={tau}$)")
    plt.grid(alpha=0.3)
    save_pdf(fig, outdir, f"qotam_spectrum_nk_E0_{E0_mid:.3f}_tau_{tau:.3f}")
    plt.show(fig)

    # Plot 2: log(n_total) vs 1/E0
    E0s = np.array([r[0] for r in rows], dtype=float)
    invE0 = np.array([r[1] for r in rows], dtype=float)
    logn  = np.array([r[3] for r in rows], dtype=float)

    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(invE0, logn, "o-")
    plt.xlabel(r"$1/E_0$")
    plt.ylabel(r"$\log n_{\rm total}$")
    plt.title(f"QOTAM non-perturbative scaling: log($n_{{total}}$) vs $1/E_0$ ($\\tau={tau}$)")
    plt.grid(alpha=0.3)
    save_pdf(fig, outdir, f"qotam_log_ntotal_vs_invE_tau_{tau:.3f}")
    plt.show(fig)

    # Plot 3: Gauge-equivalence spectrum overlay
    nk0, nkA, n0, n1, rel = gauge_equivalence_qotam(kgrid, E0_mid, tau, A0=A0, m=m, e=e)
    print("\nGauge-equivalence check (QOTAM cost depends on canonical momentum):")
    print(f"  n_total (orig)  = {n0:.6e}")
    print(f"  n_total (shift) = {n1:.6e}")
    print(f"  |Δ|/ref         = {rel:.3e}")

    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(kgrid, nk0, label="original gauge")
    #plt.plot(kgrid, nkA, "--", label=f"shifted gauge via k-> k+eA0 (A0={A0})")
    plt.plot(kgrid, nkA, "--", label=f"shifted gauge via $k \\rightarrow k + eA_0$ ($A_0$={A0})")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$n_k$ (QOTAM)")
    plt.title(f"Gauge-equivalence (QOTAM): spectrum consistency ($E_0$={E0_mid:.3f}, $\\tau={tau}$)")
    plt.grid(alpha=0.3)
    plt.legend()
    save_pdf(fig, outdir, f"qotam_gauge_equivalence_spectrum_E0_{E0_mid:.3f}_tau_{tau:.3f}_A0_{A0:.3f}")
    plt.show(fig)

    print("\nDone. QOTAM Schwinger outputs saved.")
    return outdir

# -----------------------------
# Execute
# -----------------------------
outdir = run_qotam_schwinger(
    E0_list=(0.12, 0.16, 0.22, 0.30, 0.42, 0.60),
    tau=2.5,
    m=1.0,
    e=1.0,
    Nk=600,
    kmax=7.5,
    A0=0.8
)
print("All files in:", outdir)
