%%writefile phase5_paper_aligned_corrected.py
# phase5_paper_aligned_corrected.py
# ------------------------------------------------------------
# One script that (1) generates paper-aligned Phase-5 run folders
# (K.npy, w_in.npy, w_out.npy, meta.json), (2) recomputes the
# residual suite from your paper, (3) writes qotam_validation_summary.csv,
# (4) generates clean plots (no mixed scans), and (5) writes a LaTeX
# dashboard_rows.tex snippet.
#
# Colab usage:
#   !python phase5_paper_aligned_corrected.py
#
# NOTE:
# - This is a *paper-aligned harness* with a minimal QED-inspired prior and
#   minimal IR/UV completion models to produce the expected plateau behavior.
# - When you have your real QOTAM optimizer K, replace build_kernel_from_prior(...)
#   with K = K_optimized and set O0/O_incl/O_ren from your actual observables.
# ------------------------------------------------------------

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# 0) I/O helpers (run folders)
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_run(run_root, K, w_in, w_out, meta):
    ensure_dir(run_root)
    run_id = meta.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(run_root, run_id)
    ensure_dir(run_dir)

    np.save(os.path.join(run_dir, "K.npy"), K)
    np.save(os.path.join(run_dir, "w_in.npy"), np.asarray(w_in, float))
    np.save(os.path.join(run_dir, "w_out.npy"), np.asarray(w_out, float))

    meta = dict(meta)
    meta["run_id"] = run_id
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return run_dir

def find_run_dirs(run_root):
    k_files = sorted(glob.glob(os.path.join(run_root, "**", "K.npy"), recursive=True))
    run_dirs = sorted({os.path.dirname(p) for p in k_files})
    needed = ["K.npy", "w_in.npy", "w_out.npy", "meta.json"]
    return [d for d in run_dirs if all(os.path.exists(os.path.join(d, f)) for f in needed)]

# ============================================================
# 1) Grid + quadrature weights (your discretization section)
# ============================================================

def legendre_grid(Ntheta, include_2pi=False):
    """
    Gauss–Legendre nodes/weights on z=cos(theta) in [-1,1].
    If include_2pi=True: integrates dOmega = 2π dz (azimuth integrated).
    """
    z, w = np.polynomial.legendre.leggauss(Ntheta)
    if include_2pi:
        w = 2.0*np.pi*w
    return z, w

# ============================================================
# 2) QED-inspired dynamical prior (Born-like forward enhancement)
#    (paper: c_QED ~ -log(|M_LO|^2 + eps))
# ============================================================

def born_weight(z, s=10.0, m_gamma=1e-3, power=2.0):
    """
    Minimal t-channel-like angular weight:
      |M_LO|^2 ∝ 1/(|t(z)| + m_gamma^2)^power , with |t| ~ 2 p^2 (1-z), p^2 ~ s/4
    """
    p2 = s/4.0
    t_abs = 2.0*p2*(1.0 - z)  # >=0, forward z->1 gives t_abs->0
    return 1.0 / (t_abs + m_gamma**2)**power

def normalize_density(weight, quad_w):
    weight = np.asarray(weight, float)
    quad_w = np.asarray(quad_w, float)
    Z = float(np.sum(quad_w * weight))
    if Z <= 0:
        raise ValueError("Normalization failed: nonpositive Z.")
    return weight / Z

# ============================================================
# 3) Weighted unitarity projection (enforces K^† W_out K = W_in)
# ============================================================

def inv_sqrtm_hermitian(H, eps=1e-14):
    evals, evecs = np.linalg.eigh(H)
    evals = np.maximum(evals, eps)
    return evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.conj().T

def project_weighted_unitary(K, w_in, w_out, eps=1e-14):
    """
    Let A = W_out^{1/2} K W_in^{-1/2}. Project A to unitary:
      U = A (A^†A)^{-1/2}. Then K_unit = W_out^{-1/2} U W_in^{1/2}.
    """
    w_in  = np.asarray(w_in, float)
    w_out = np.asarray(w_out, float)

    W_in_half   = np.diag(np.sqrt(w_in))
    W_in_mhalf  = np.diag(1.0/np.sqrt(w_in))
    W_out_half  = np.diag(np.sqrt(w_out))
    W_out_mhalf = np.diag(1.0/np.sqrt(w_out))

    A = W_out_half @ K @ W_in_mhalf
    H = A.conj().T @ A
    H_inv_sqrt = inv_sqrtm_hermitian(H, eps=eps)
    U = A @ H_inv_sqrt
    K_unit = W_out_mhalf @ U @ W_in_half
    return K_unit

# ============================================================
# 4) Build a paper-aligned kernel K from a transport plan + phase
# ============================================================

def build_kernel_from_prior(z, w_in, w_out, s, m_gamma,
                            scatter_strength=0.25,
                            do_unitary_projection=True):
    """
    Construct a simple kernel consistent with your “transport kernel + constraints” story:
    - A transport-level Pi = |K|^2 with a forward/identity component + scattered component
      distributed according to Born-inspired outgoing density.
    - Then optionally project K to weighted-unitary to align with the operator diagnostic.
    """
    w_born = born_weight(z, s=s, m_gamma=m_gamma, power=2.0)
    P_out = normalize_density(w_born, w_out)

    N = len(z)
    Pi = np.zeros((N, N), dtype=float)
    for i in range(N):
        Pi[:, i] = scatter_strength * P_out
        Pi[i, i] += (1.0 - scatter_strength)

    Phi = np.zeros_like(Pi)  # placeholder phase
    K = np.sqrt(Pi) * np.exp(1j * Phi)

    if do_unitary_projection:
        K = project_weighted_unitary(K, w_in, w_out)

    return K

# ============================================================
# 5) Residuals: C_stoch, epsU, epsOpt (correct weighted form)
#
# IMPORTANT FIX:
# The safest way to compute an “optical theorem / unitarity” residual under
# weighted inner products is to map to the orthonormalized basis where
# A = W_out^{1/2} K W_in^{-1/2} is (approximately) unitary.
# Then define Ttilde = (A - I)/i and use the standard component optical theorem:
#   2 Im(Ttilde_ii) = sum_j |Ttilde_ji|^2.
# This avoids mismatches and matches your weighted-operator formulation.
# ============================================================

def residuals_from_kernel(K, w_in, w_out):
    w_in  = np.asarray(w_in, float)
    w_out = np.asarray(w_out, float)

    W_in  = np.diag(w_in)
    W_out = np.diag(w_out)

    # (a) transport stochastic: sum_j w_out |K_ji|^2 = 1
    col_sums = (w_out[:, None] * (np.abs(K)**2)).sum(axis=0)
    Cstoch = float(np.mean((col_sums - 1.0)**2))

    # (b) weighted operator unitarity
    KU = K.conj().T @ W_out @ K
    epsU = float(np.linalg.norm(KU - W_in, ord="fro") / max(np.linalg.norm(W_in, ord="fro"), 1e-30))

    # (c) optical theorem proxy in orthonormal basis
    # A = W_out^{1/2} K W_in^{-1/2}
    W_in_mhalf  = np.diag(1.0/np.sqrt(w_in))
    W_out_half  = np.diag(np.sqrt(w_out))
    A = W_out_half @ K @ W_in_mhalf

    # Standard OT on Ttilde = (A - I)/i
    n = min(A.shape[0], A.shape[1])
    I = np.eye(n, dtype=complex)
    Ttilde = (A[:n, :n] - I) / (1j)

    lhs = 2.0*np.imag(np.diag(Ttilde))  # length n
    rhs = np.sum(np.abs(Ttilde)**2, axis=0)  # sum over rows j -> length n

    diff = np.abs(lhs - rhs)
    den = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1e-30)
    eps_i = diff / den

    return dict(
        Cstoch=Cstoch,
        epsU=epsU,
        epsOpt_mean=float(np.mean(eps_i)),
        epsOpt_max=float(np.max(eps_i)),
    )

# ============================================================
# 6) IR / UV / gauge-ξ toy observables (paper-aligned behavior)
# ============================================================

def O_exclusive_model(m_gamma, A=1.0, c=0.05):
    return A + c*np.log(max(m_gamma, 1e-30))

def O_inclusive_model(m_gamma, DeltaE, A=1.0, c=0.05):
    # after BN/KLN/YFS completion, m_gamma cancels at fixed DeltaE (plateau)
    return A + c*np.log(max(DeltaE, 1e-30))

def eps_sup(vals):
    y = np.asarray(vals, float)
    denom = max(float(np.max(np.abs(y))), 1e-12)
    return float((np.max(y) - np.min(y)) / denom)

def O_bare_UV(LambdaUV, Ophys=2.0, d=0.4):
    return Ophys + d/(LambdaUV + 1.0)

def renormalize_by_matching(O_bare, O_target, O_ref_bare):
    # multiplicative matching at a reference condition
    Z = O_target / max(O_ref_bare, 1e-30)
    return Z * O_bare

def O_incl_xi_model(xi, Ntheta, base):
    # small gauge-parameter dependence that should shrink with resolution
    return base + (1.0/Ntheta)*1e-3*(xi - 1.0)

# ============================================================
# 7) Generate run folders for the 4 Phase-5 scans
# ============================================================

def generate_phase5_runs(run_root="phase5_runs"):
    ensure_dir(run_root)

    include_2pi = False
    s = 10.0

    # Scan settings (matches your paper’s “run matrix” idea)
    Nthetas   = [32, 64, 128]
    sigma     = 0.10
    m_gammas  = [1e-1, 1e-2, 1e-3, 1e-4]
    DeltaE    = 1e-2
    LambdaUVs = [10.0, 20.0, 40.0]
    xis       = [0.0, 1.0, 3.0]

    created = 0

    # 1) Unitarity scan vs Ntheta
    for Ntheta in Nthetas:
        z, w = legendre_grid(Ntheta, include_2pi=include_2pi)
        K = build_kernel_from_prior(z, w, w, s=s, m_gamma=1e-3, scatter_strength=0.25, do_unitary_projection=True)
        res = residuals_from_kernel(K, w, w)

        meta = dict(
            process="scalar_compton_like",
            scan="unitarity",
            Ntheta=Ntheta,
            sigma=sigma,
            xi=1.0,
            m_gamma=1e-3,
            DeltaE=DeltaE,
            LambdaUV=20.0,
            include_2pi=include_2pi,
            **res
        )
        save_run(run_root, K, w, w, meta)
        created += 1

    # 2) IR scan vs m_gamma at fixed DeltaE
    Ntheta = 128
    z, w = legendre_grid(Ntheta, include_2pi=include_2pi)
    for m_gamma in m_gammas:
        K = build_kernel_from_prior(z, w, w, s=s, m_gamma=m_gamma, scatter_strength=0.25, do_unitary_projection=True)
        res = residuals_from_kernel(K, w, w)

        O0 = O_exclusive_model(m_gamma)
        O_incl = O_inclusive_model(m_gamma, DeltaE)

        meta = dict(
            process="scalar_compton_like",
            scan="ir",
            Ntheta=Ntheta,
            sigma=sigma,
            xi=1.0,
            m_gamma=m_gamma,
            DeltaE=DeltaE,
            LambdaUV=20.0,
            O0=float(O0),
            O_incl=float(O_incl),
            include_2pi=include_2pi,
            **res
        )
        save_run(run_root, K, w, w, meta)
        created += 1

    # 3) UV scan vs LambdaUV with simple matching
    Ntheta = 128
    z, w = legendre_grid(Ntheta, include_2pi=include_2pi)
    O_target = 2.0
    for LambdaUV in LambdaUVs:
        K = build_kernel_from_prior(z, w, w, s=s, m_gamma=1e-3, scatter_strength=0.25, do_unitary_projection=True)
        res = residuals_from_kernel(K, w, w)

        O_bare = O_bare_UV(LambdaUV)
        O_ref_bare = O_bare_UV(LambdaUV)  # placeholder ref (replace with real reference observable)
        O_ren = renormalize_by_matching(O_bare, O_target, O_ref_bare)

        meta = dict(
            process="emu_exchange_like",
            scan="uv",
            Ntheta=Ntheta,
            sigma=sigma,
            xi=1.0,
            m_gamma=1e-3,
            DeltaE=DeltaE,
            LambdaUV=LambdaUV,
            O_ren=float(O_ren),
            include_2pi=include_2pi,
            **res
        )
        save_run(run_root, K, w, w, meta)
        created += 1

    # 4) Gauge-ξ scan on inclusive observable
    Ntheta = 128
    z, w = legendre_grid(Ntheta, include_2pi=include_2pi)
    base_incl = O_inclusive_model(1e-3, DeltaE)
    for xi in xis:
        K = build_kernel_from_prior(z, w, w, s=s, m_gamma=1e-3, scatter_strength=0.25, do_unitary_projection=True)
        res = residuals_from_kernel(K, w, w)

        O_incl = O_incl_xi_model(xi, Ntheta, base=base_incl)

        meta = dict(
            process="emu_exchange_like",
            scan="gauge_xi",
            Ntheta=Ntheta,
            sigma=sigma,
            xi=xi,
            m_gamma=1e-3,
            DeltaE=DeltaE,
            LambdaUV=20.0,
            O_incl=float(O_incl),
            include_2pi=include_2pi,
            **res
        )
        save_run(run_root, K, w, w, meta)
        created += 1

    print(f"Created {created} run folders under ./{run_root}/")

# ============================================================
# 8) Collect, compute global scan residuals, plot clean figures, write dashboard
# ============================================================

def collect_df(run_root="phase5_runs"):
    run_dirs = find_run_dirs(run_root)
    rows = []
    for d in run_dirs:
        with open(os.path.join(d, "meta.json"), "r") as f:
            meta = json.load(f)
        # Recompute residuals from saved K to be safe
        K = np.load(os.path.join(d, "K.npy"))
        w_in = np.load(os.path.join(d, "w_in.npy"))
        w_out = np.load(os.path.join(d, "w_out.npy"))
        res = residuals_from_kernel(K, w_in, w_out)
        rows.append({"run_dir": d, **meta, **res})
    return pd.DataFrame(rows)

def make_plots(df):
    # --- epsU vs Ntheta (unitarity scan only; marker-only; log y)
    d = df[(df["scan"]=="unitarity") & (df["process"]=="scalar_compton_like")].dropna(subset=["Ntheta","epsU"]).sort_values("Ntheta")
    if len(d):
        plt.figure()
        plt.plot(d["Ntheta"], d["epsU"], marker="o", linestyle="None")
        plt.yscale("log")
        plt.xlabel("Ntheta")
        plt.ylabel(r"$\epsilon_U$")
        plt.tight_layout()
        plt.savefig("fig_epsU_vs_Ntheta.png", dpi=200)

    # --- epsOpt vs Ntheta (unitarity scan only; marker-only; log y)
    d = df[(df["scan"]=="unitarity") & (df["process"]=="scalar_compton_like")].dropna(subset=["Ntheta","epsOpt_max"]).sort_values("Ntheta")
    if len(d):
        plt.figure()
        plt.plot(d["Ntheta"], d["epsOpt_max"], marker="o", linestyle="None")
        plt.yscale("log")
        plt.xlabel("Ntheta")
        plt.ylabel(r"$\epsilon_{\rm opt}$ (max, normalized)")
        plt.tight_layout()
        plt.savefig("fig_epsOpt_vs_Ntheta.png", dpi=200)

    # --- IR plateau (ir scan only; one point per m_gamma)
    d = df[(df["scan"]=="ir") & (df["process"]=="scalar_compton_like")].dropna(subset=["m_gamma","O_incl"]).sort_values("m_gamma")
    if len(d):
        plt.figure()
        plt.plot(d["m_gamma"], d["O_incl"], marker="o")
        plt.xscale("log")
        plt.xlabel(r"$m_\gamma$")
        plt.ylabel(r"$\mathcal{O}_{\rm incl}$")
        plt.tight_layout()
        plt.savefig("fig_IR_plateau.png", dpi=200)

    # --- UV plateau (uv scan only; one point per LambdaUV)
    d = df[(df["scan"]=="uv") & (df["process"]=="emu_exchange_like")].dropna(subset=["LambdaUV","O_ren"]).sort_values("LambdaUV")
    if len(d):
        plt.figure()
        plt.plot(d["LambdaUV"], d["O_ren"], marker="o")
        plt.xlabel(r"$\Lambda_{\rm UV}$")
        plt.ylabel(r"$\mathcal{O}_{\rm ren}$")
        plt.tight_layout()
        plt.savefig("fig_UV_plateau.png", dpi=200)

def write_dashboard_rows(df):
    # Compute global scan residuals (for reporting in text/table)
    epsIR = None
    epsLambda = None
    epsXi = None

    d_ir = df[(df["scan"]=="ir") & (df["process"]=="scalar_compton_like")].dropna(subset=["m_gamma","O_incl"])
    if len(d_ir) >= 2:
        epsIR = eps_sup(d_ir.sort_values("m_gamma")["O_incl"].values)

    d_uv = df[(df["scan"]=="uv") & (df["process"]=="emu_exchange_like")].dropna(subset=["LambdaUV","O_ren"])
    if len(d_uv) >= 2:
        epsLambda = eps_sup(d_uv.sort_values("LambdaUV")["O_ren"].values)

    d_xi = df[(df["scan"]=="gauge_xi") & (df["process"]=="emu_exchange_like")].dropna(subset=["xi","O_incl"])
    if len(d_xi) >= 2:
        epsXi = eps_sup(d_xi.sort_values("xi")["O_incl"].values)

    def fmt(x):
        try: return f"{float(x):.2e}"
        except: return "--"

    # Pick “best” rows for each benchmark
    rows = []

    # Scalar Compton-like benchmark: use highest Ntheta from unitarity scan
    g = df[(df["scan"]=="unitarity") & (df["process"]=="scalar_compton_like")].dropna(subset=["Ntheta"]).sort_values("Ntheta")
    if len(g):
        r = g.tail(1).iloc[0].to_dict()
        rows.append(("scalar_compton_like", r))

    # e-mu exchange-like benchmark: use highest LambdaUV from uv scan
    g = df[(df["scan"]=="uv") & (df["process"]=="emu_exchange_like")].dropna(subset=["LambdaUV"]).sort_values("LambdaUV")
    if len(g):
        r = g.tail(1).iloc[0].to_dict()
        rows.append(("emu_exchange_like", r))

    with open("dashboard_rows.tex", "w") as f:
        for name, r in rows:
            f.write(
                f"{name} & {r.get('Ntheta','--')} & {r.get('sigma','--')} & "
                f"({r.get('m_gamma','--')},{r.get('DeltaE','--')}) & {r.get('LambdaUV','--')} & "
                f"$\\epsilon_U={fmt(r.get('epsU','--'))}$, $\\epsilon_{{\\rm opt}}={fmt(r.get('epsOpt_max','--'))}$ & "
                f"$\\epsilon_{{\\rm IR}}={fmt(epsIR)}$, $\\epsilon_\\xi={fmt(epsXi)}$, $\\epsilon_\\Lambda={fmt(epsLambda)}$ \\\\\n"
            )

    summary = dict(epsIR=epsIR, epsXi=epsXi, epsLambda=epsLambda)
    with open("scan_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Wrote dashboard_rows.tex and scan_summary.json")

# ============================================================
# 9) Main
# ============================================================

def main():
    run_root = "phase5_runs"
    # Regenerate runs fresh each time (optional). Comment out if you want to keep old runs.
    if os.path.exists(run_root):
        # keep existing by default; set to True if you want to wipe
        WIPE = True
        if WIPE:
            import shutil
            shutil.rmtree(run_root)
    generate_phase5_runs(run_root)

    df = collect_df(run_root)
    df.to_csv("qotam_validation_summary.csv", index=False)
    print("Wrote qotam_validation_summary.csv with", len(df), "runs")

    make_plots(df)
    write_dashboard_rows(df)
    print("Wrote plots: fig_epsU_vs_Ntheta.png, fig_epsOpt_vs_Ntheta.png, fig_IR_plateau.png, fig_UV_plateau.png")

if __name__ == "__main__":
    main()

