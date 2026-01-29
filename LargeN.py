import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
import matplotlib.ticker as mticker

DELTA_FLOOR = 1e-12
NBINS = 20
SCATTER = 0.25
S = 10.0
M_GAMMA = 1e-3

Z0_IN = -0.2
SIGMA_Z = 0.25

def born_weight(z, s=10.0, m_gamma=1e-3, power=2.0):
    p2 = s/4.0
    t_abs = 2.0*p2*(1.0 - z)           # |t| ~ 2 p^2 (1-z)
    return 1.0 / (t_abs + m_gamma**2)**power

def normalize_density(weight, w):
    Z = float(np.sum(w * weight))
    return weight / max(Z, 1e-300)

def build_Pi(N, scatter=0.25, s=10.0, m_gamma=1e-3):
    z, w = leggauss(N)
    P_out = normalize_density(born_weight(z, s=s, m_gamma=m_gamma), w)
    # Correct weighted stochastic column sums:
    # sum_j w_j Pi_{ji} = scatter*sum_j w_j P_out[j] + (1-scatter) = 1
    Pi = scatter * P_out[:, None] * np.ones((1, N))
    Pi[np.arange(N), np.arange(N)] += (1.0 - scatter) / w
    return z, w, Pi

def gaussian_wp(z, z0=-0.2, sigma=0.25):
    return np.exp(-0.5*((z - z0)/sigma)**2)

def normalize_weighted(psi, w):
    return psi / max(np.sqrt(np.sum(w*np.abs(psi)**2)), 1e-300)

def out_bins_from_Pi(Pi, z, w_in, w_out, nbins=20, z0=-0.2, sigma=0.25):
    # rho_in(p) = |psi_in(p)|^2 with ∫ dp |psi|^2 = 1
    psi = normalize_weighted(gaussian_wp(z, z0=z0, sigma=sigma), w_in)
    rho_in = np.abs(psi)**2

    # Discrete pushforward:
    # rho_out[j] = sum_i w_in[i] Pi[j,i] rho_in[i]
    rho_out = Pi @ (w_in * rho_in)

    # Bin probability mass: P_b = sum_{j in bin} w_out[j] rho_out[j]
    edges = np.linspace(-1.0, 1.0, nbins+1)
    Pbins = np.zeros(nbins)
    for b in range(nbins):
        lo, hi = edges[b], edges[b+1]
        if b < nbins-1:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)
        Pbins[b] = float(np.sum(w_out[mask] * rho_out[mask]))
    return Pbins

def delta_conv(P2, P1, delta_floor=1e-12):
    denom = max(np.sum(np.abs(P2)), delta_floor)  # ~1
    return float(np.sum(np.abs(P2 - P1)) / denom)

# Choose very large N values
#Ns = [ 32, 64, 128, 256, 512, 1024, 2048, 4096]
Ns = [ 24, 44, 56, 60, 128, 256, 512, 1024, 2048, 4096]

# Compute binned distributions for each N
P_by = []
for N in Ns:
    z, w, Pi = build_Pi(N, scatter=SCATTER, s=S, m_gamma=M_GAMMA)
    Pbins = out_bins_from_Pi(Pi, z, w, w, nbins=NBINS, z0=Z0_IN, sigma=SIGMA_Z)
    P_by.append((N, Pbins))

# Compute DeltaConv between successive refinements
N_plot, Delta_plot = [], []
for i in range(len(P_by)-1):
    N1, P1 = P_by[i]
    N2, P2 = P_by[i+1]
    N_plot.append(N2)
    Delta_plot.append(delta_conv(P2, P1, delta_floor=DELTA_FLOOR))

print("DeltaConv points (Ntheta -> DeltaConv):")
for n, d in zip(N_plot, Delta_plot):
    print(n, "->", d)

# Plot to PDF
plt.figure()
plt.plot(N_plot, Delta_plot, marker="o", linestyle="None")
plt.yscale("log")
plt.xlabel(r"$N_\theta$")
plt.ylabel(r"$\Delta_{\rm conv}$ (binned $L^1$, $\Pi$-pushforward)")

plt.tight_layout()
plt.savefig("fig_DeltaConv_binned_largeN.pdf", bbox_inches="tight")
print("Wrote fig_DeltaConv_binned_largeN.pdf")
