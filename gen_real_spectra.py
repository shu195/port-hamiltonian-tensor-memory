import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ph_ssm import PortHamiltonianSSM, compute_effective_rank

# Dark-mode publication styling matching generate_visuals.py
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#222222'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(REPO_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def compute_ridge_r2(h_k: torch.Tensor, h_l: torch.Tensor, lam: float = 1e-4) -> float:
    """Eq. (12) in paper/main.tex: ridge-regularized R^2(h_k | h_l)."""
    D = h_l.shape[1]
    eye = torch.eye(D, device=h_l.device, dtype=h_l.dtype)
    W_star = torch.linalg.solve(h_l.T @ h_l + lam * eye, h_l.T @ h_k)
    pred = h_l @ W_star
    ss_res = torch.sum((h_k - pred) ** 2)
    ss_tot = torch.sum((h_k - h_k.mean(dim=0, keepdim=True)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    return max(0.0, r2.item())


def compute_real_spectral_audit():
    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------------------------------------------------
    # Panel (a): Singular Value & Effective Rank Spectrum (D=64)
    # ---------------------------------------------------------
    B, L, D = 256, 16, 64
    z0 = torch.randn(B, D)
    u = 0.20 * torch.randn(B, L, D)

    # 1. Real PortHamiltonianSSM (D=64)
    ph_64 = PortHamiltonianSSM(d_model=D, dt=0.02)
    with torch.no_grad():
        _, ph_states = ph_64(u, z0=z0, return_states=True)
    h_ph = ph_states[:, -1, :]
    s_ph = torch.linalg.svdvals(h_ph).numpy()
    er_ph = compute_effective_rank(h_ph)

    # 2. Real Deep ResNet baseline (16 residual layers, D=64)
    resnet_layers = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(L)])
    for layer in resnet_layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.35 / math.sqrt(D))
    with torch.no_grad():
        h_res = z0.clone()
        for t in range(L):
            h_res = h_res + 0.5 * torch.tanh(resnet_layers[t](h_res)) + 0.02 * u[:, t, :]
    s_res = torch.linalg.svdvals(h_res).numpy()
    er_res = compute_effective_rank(h_res)

    # 3. Real Isotropic Damped baseline (constant isotropic decay alpha=0.25, no skew J)
    with torch.no_grad():
        h_iso = z0.clone()
        W_iso = torch.randn(D, D) * (0.15 / math.sqrt(D))
        for t in range(L):
            h_iso = 0.78 * h_iso + 0.10 * torch.tanh(h_iso @ W_iso) + 0.02 * u[:, t, :]
    s_iso = torch.linalg.svdvals(h_iso).numpy()
    er_iso = compute_effective_rank(h_iso)

    print("=== PANEL (a): D=64 SINGULAR VALUE & ERANK SPECTRUM ===")
    print(f"PH-SSM (D=64):          erank = {er_ph:.4f} / {D} ({100*er_ph/D:.2f}%)")
    print(f"Deep ResNet (D=64):     erank = {er_res:.4f} / {D} ({100*er_res/D:.2f}%)")
    print(f"Isotropic Damped (D=64):erank = {er_iso:.4f} / {D} ({100*er_iso/D:.2f}%)")

    # ---------------------------------------------------------
    # Panel (b): Complex Eigenvalue Spectrum in C (D=32)
    # ---------------------------------------------------------
    ph_32 = PortHamiltonianSSM(d_model=32, dt=0.02, gamma_0=0.15, h_target=1.0, tau=0.5)
    with torch.no_grad():
        A_J = ph_32.get_cayley_operator()
        eig_skew = torch.linalg.eigvals(A_J).numpy()
        det_AJ = torch.linalg.det(A_J).item()
        mags_AJ = np.abs(eig_skew)

        # Burst radial damping Jacobian scaling at high-energy burst (H(z) = 10.0 >> h_target)
        z_burst = torch.ones(1, 32) * math.sqrt(2.0 * 10.0 / 32.0)
        z_after = ph_32.dissipation_half_step(
            ph_32.dissipation_half_step(z_burst, 0.5 * ph_32.dt),
            0.5 * ph_32.dt,
        )
        burst_scale = (torch.norm(z_after) / torch.norm(z_burst)).item()
        eig_ph_burst = eig_skew * burst_scale

        # Higher-dt burst regime from real PortHamiltonianSSM(dt=0.25, gamma_0=0.6) for visual clarity
        ph_32_vis = PortHamiltonianSSM(d_model=32, dt=0.25, gamma_0=0.6, h_target=1.0, tau=0.5)
        A_J_vis = ph_32_vis.get_cayley_operator()
        eig_skew_vis = torch.linalg.eigvals(A_J_vis).numpy()
        z_after_vis = ph_32_vis.dissipation_half_step(
            ph_32_vis.dissipation_half_step(z_burst, 0.5 * ph_32_vis.dt),
            0.5 * ph_32_vis.dt,
        )
        burst_scale_vis = (torch.norm(z_after_vis) / torch.norm(z_burst)).item()
        eig_ph_burst_vis = eig_skew_vis * burst_scale_vis

        # Isotropic attenuation baseline (gamma_iso = 0.28)
        eig_iso = eig_skew * 0.72

    print("\n=== PANEL (b): D=32 COMPLEX EIGENVALUE SPECTRUM ===")
    print(f"A_J (dt=0.02): det(A_J) = {det_AJ:.10f}, |lam|_max = {mags_AJ.max():.10f}, |lam|_min = {mags_AJ.min():.10f}, max_unit_err = {np.abs(mags_AJ - 1.0).max():.3e}")
    print(f"Burst radial damping (dt=0.02, gamma_0=0.15, H=10): contraction = {burst_scale:.6f}")
    print(f"Burst radial damping (dt=0.25, gamma_0=0.60, H=10): contraction = {burst_scale_vis:.6f}")

    # ---------------------------------------------------------
    # Panel (c): Inter-Layer True Tensor Memory M(k, l) (Eq. true_tensor_memory)
    # ---------------------------------------------------------
    # Evaluate across 16 steps/layers of real PortHamiltonianSSM (D=64, dt=0.10)
    ph_mem = PortHamiltonianSSM(d_model=D, dt=0.10)
    with torch.no_grad():
        _, mem_states = ph_mem(0.25 * torch.randn(B, L, D), z0=torch.randn(B, D), return_states=True)
    # mem_states shape: [B, 16, D]
    M_matrix = np.zeros((L, L))
    step_arcs = []
    for tau in range(L - 1):
        num = torch.norm(mem_states[:, tau + 1, :] - mem_states[:, tau, :], dim=-1).mean().item()
        den = torch.norm(mem_states[:, tau, :], dim=-1).mean().item()
        step_arcs.append(num / (den + 1e-12))

    eranks_by_layer = [compute_effective_rank(mem_states[:, idx, :]) for idx in range(L)]
    for k in range(L):
        for l in range(L):
            if l > k:
                r2_kl = compute_ridge_r2(mem_states[:, k, :], mem_states[:, l, :], lam=1e-4)
                rank_factor = min(eranks_by_layer[k], eranks_by_layer[l]) / D
                w_kl = sum(step_arcs[k:l])
                work_factor = 1.0 - math.exp(-w_kl)
                M_matrix[k, l] = r2_kl * rank_factor * work_factor

    r2_0_15 = compute_ridge_r2(mem_states[:, 0, :], mem_states[:, 15, :], lam=1e-4)
    w_0_15 = sum(step_arcs[0:15])
    print("\n=== PANEL (c): TRUE TENSOR MEMORY M(k, l) (D=64, L=16, dt=0.10) ===")
    print(f"erank(h_0)  = {eranks_by_layer[0]:.4f} / {D}")
    print(f"erank(h_15) = {eranks_by_layer[15]:.4f} / {D}")
    print(f"R^2(h_0 | h_15) = {r2_0_15:.6f}")
    print(f"W(0, 15)        = {w_0_15:.6f} (1 - exp(-W) = {1.0 - math.exp(-w_0_15):.6f})")
    print(f"M(0, 15)        = {M_matrix[0, 15]:.6f}")
    print(f"M_matrix max    = {M_matrix.max():.6f}")

    # ---------------------------------------------------------
    # Render Figure 5: assets/spectral_audit_trio.png
    # ---------------------------------------------------------
    k_vals = np.arange(1, D + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15.5, 4.6), dpi=300,
        gridspec_kw={'width_ratios': [1.15, 1.0, 1.15]}
    )

    # Panel (a)
    ax1.plot(k_vals, s_ph / s_ph[0], color='#00e676', linewidth=2.2,
             label=f'PH-SSM (erank={er_ph:.1f}/64)')
    ax1.plot(k_vals, s_res / s_res[0], color='#00b0ff', linewidth=1.9,
             label=f'Deep ResNet (erank={er_res:.1f}/64)')
    ax1.plot(k_vals, s_iso / s_iso[0], color='#ff3366', linewidth=1.9, linestyle='--',
             label=f'Isotropic Damped (erank={er_iso:.1f}/64)')
    ax1.set_title("(a) Singular Value & Effective Rank Spectrum", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Singular Value Index k", fontsize=10.5, color='#cccccc')
    ax1.set_ylabel(r"Normalized $\sigma_k / \sigma_1$ (log scale)", fontsize=10.5, color='#cccccc')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor='#111111', edgecolor='#444444', fontsize=8.8, loc='lower left')

    # Panel (b)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color='#ffffff', linestyle='--', linewidth=1.1,
             label=r'Unit Circle $|\lambda|=1$')
    ax2.scatter(eig_skew.real, eig_skew.imag, color='#ff4444', s=42, alpha=0.85,
                edgecolors='#ffffff', linewidth=0.5, label='Equilibrium SO(32) (dt=0.02)')
    ax2.scatter(eig_ph_burst_vis.real, eig_ph_burst_vis.imag, color='#00e676', s=46,
                edgecolors='#ffffff', linewidth=0.6, label=f'Burst Radial Damping (|λ|={burst_scale_vis:.2f})')
    ax2.scatter(eig_iso.real, eig_iso.imag, color='#ffbb33', s=36, marker='x',
                label='Isotropic Attenuation (|λ|=0.72)')
    ax2.axhline(0, color='#333333', linewidth=0.8)
    ax2.axvline(0, color='#333333', linewidth=0.8)
    ax2.set_title(r"(b) Complex Spectrum in $\mathbb{C}$", fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xlabel(r"$\mathrm{Re}(\lambda)$", fontsize=10.5, color='#cccccc')
    ax2.set_ylabel(r"$\mathrm{Im}(\lambda)$", fontsize=10.5, color='#cccccc')
    ax2.set_xlim(-1.22, 1.22)
    ax2.set_ylim(-1.22, 1.22)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor='#111111', edgecolor='#444444', fontsize=7.6, loc='upper left')

    # Panel (c)
    vmax = max(0.85, float(M_matrix.max()) * 1.05)
    cax = ax3.imshow(M_matrix, cmap='viridis', origin='lower', aspect='auto', vmin=0.0, vmax=vmax)
    ax3.set_title(r"(c) True Tensor Memory $\mathcal{M}(k, l)$", fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xlabel("Target Layer l", fontsize=10.5, color='#cccccc')
    ax3.set_ylabel("Source Layer k", fontsize=10.5, color='#cccccc')
    cbar = fig.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label(r"Capacity $\mathcal{M}(k, l)$", fontsize=9.5, color='#cccccc')
    m_0_15 = M_matrix[0, 15]
    ax3.annotate(
        rf'$\mathcal{{M}}(0, 15) = {m_0_15:.3f}$',
        xy=(15, 0), xytext=(5.0, 4.2),
        arrowprops=dict(facecolor='#00e676', shrink=0.08, width=1.2, headwidth=5.5),
        fontsize=9.2, color='#00e676', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', edgecolor='#00e676', alpha=0.9)
    )

    plt.tight_layout()
    trio_path = os.path.join(ASSETS_DIR, "spectral_audit_trio.png")
    plt.savefig(trio_path, dpi=300, facecolor='#000000', edgecolor='none')
    plt.close()
    print(f"\nGenerated: {trio_path}")


if __name__ == "__main__":
    compute_real_spectral_audit()
