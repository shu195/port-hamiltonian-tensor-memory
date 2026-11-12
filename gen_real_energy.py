import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from ph_ssm import PortHamiltonianSSM

# Styling for dark-mode publication aesthetics (matching generate_visuals.py)
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


def run_isotropic_damped_baseline(
    model_ref: PortHamiltonianSSM,
    u: torch.Tensor,
    z0: torch.Tensor,
    gamma_iso: float = 0.5,
) -> torch.Tensor:
    """
    Runs an isotropic-damped baseline sharing the exact same Cayley operator A_J
    and input gate (W_gate, b_gate) as model_ref, but replacing state-dependent
    radial Rayleigh damping with constant isotropic damping r = gamma_iso.
    """
    B, L, _ = u.shape
    A_J = model_ref.get_cayley_operator()
    half_dt = 0.5 * model_ref.dt
    z = z0.clone()
    states = []
    for k in range(L):
        u_k = u[:, k, :]
        gate = torch.sigmoid(torch.matmul(z, model_ref.W_gate) + model_ref.b_gate)
        u_gated = gate * u_k
        z_half = z / (1.0 + half_dt * gamma_iso)
        z_rot = torch.matmul(z_half, A_J.T) + model_ref.dt * u_gated
        z = z_rot / (1.0 + half_dt * gamma_iso)
        states.append(z)
    return torch.stack(states, dim=1)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    B, L, D = 16, 1000, 32
    dt = 0.02
    burst_start, burst_end = 200, 700
    burst_sigma = 10.0
    gamma_iso = 0.5

    # Real PortHamiltonianSSM (default parameters: gamma_0=0.15, h_target=1.0, tau=0.5)
    model_ph = PortHamiltonianSSM(d_model=D, dt=dt)
    model_ph.eval()

    # Pure symplectic baseline (R = 0): identical weights, gamma_0 = 0.0
    model_symp = PortHamiltonianSSM(d_model=D, dt=dt, gamma_0=0.0)
    model_symp.load_state_dict(model_ph.state_dict())
    model_symp.eval()

    # Initial state on equilibrium manifold H(z_0) = 0.5 ||z_0||^2 = 1.0
    z0 = torch.randn(B, D)
    z0 = z0 * math.sqrt(2.0) / torch.norm(z0, dim=-1, keepdim=True)

    # 1000-step input with high-energy Gaussian distractor burst on [200, 700]
    u = torch.zeros(B, L, D)
    u[:, burst_start:burst_end, :] = burst_sigma * torch.randn(B, burst_end - burst_start, D)

    with torch.no_grad():
        _, states_ph = model_ph(u, z0=z0, return_states=True)
        _, states_symp = model_symp(u, z0=z0, return_states=True)
        states_iso = run_isotropic_damped_baseline(model_ph, u, z0, gamma_iso=gamma_iso)

    # Measured Hamiltonian energy H(z_t) = 0.5 * ||z_t||_2^2 (batch mean)
    e_ph = (0.5 * torch.sum(states_ph ** 2, dim=-1)).mean(dim=0).numpy()
    e_symp = (0.5 * torch.sum(states_symp ** 2, dim=-1)).mean(dim=0).numpy()
    e_iso = (0.5 * torch.sum(states_iso ** 2, dim=-1)).mean(dim=0).numpy()

    symp_peak, symp_final = float(e_symp.max()), float(e_symp[-1])
    ph_peak, ph_final = float(e_ph.max()), float(e_ph[-1])
    iso_peak, iso_final = float(e_iso.max()), float(e_iso[-1])

    print("=" * 68)
    print(f"REAL PH-SSM ENERGY TRAJECTORY AUDIT (B={B}, L={L}, D={D}, dt={dt})")
    print(f"Distractor burst: [{burst_start}, {burst_end}), sigma={burst_sigma}")
    print("-" * 68)
    print(f"Pure Symplectic (R=0, gamma_0=0.0):   Peak H = {symp_peak:.4f}, Final H = {symp_final:.4f}")
    print(f"PH-SSM (Real, gamma_0={model_ph.gamma_0}):         Peak H = {ph_peak:.4f}, Final H = {ph_final:.4f}")
    print(f"Isotropic Damped (gamma_iso={gamma_iso}):     Peak H = {iso_peak:.4f}, Final H = {iso_final:.4f}")
    print("=" * 68)

    steps = np.arange(L)
    y_max = max(symp_peak, ph_peak, iso_peak) * 1.15

    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=300)
    ax.fill_between(
        [burst_start, burst_end], 0, y_max,
        color='#ff3366', alpha=0.12,
        label=f'Distractor Noise Burst Horizon (t ∈ [{burst_start}, {burst_end}])'
    )

    ax.plot(
        steps, e_symp, color='#ff4444', linewidth=2.0,
        label=f'Pure Symplectic (R = 0: Peak H = {symp_peak:.2f}, Final H = {symp_final:.2f})'
    )
    ax.plot(
        steps, e_ph, color='#00e676', linewidth=2.3,
        label=f'PH-SSM (Real, γ₀={model_ph.gamma_0}: Peak H = {ph_peak:.2f}, Final H = {ph_final:.3f})'
    )
    ax.plot(
        steps, e_iso, color='#ffbb33', linewidth=2.0, linestyle=':',
        label=f'Isotropic Damped (γ_iso={gamma_iso}: Peak H = {iso_peak:.2f}, Final H = {iso_final:.3f})'
    )
    ax.axhline(
        model_ph.h_target, color='#888888', linestyle='--', linewidth=1.0, alpha=0.7,
        label=f'Target Equilibrium Manifold (H_target = {model_ph.h_target:.1f})'
    )

    ax.set_title(
        "Measured Hamiltonian Energy Trajectory H(z_t) Under Distractor Bursts",
        fontsize=12.5, fontweight='bold', pad=10, color='#ffffff'
    )
    ax.set_xlabel("Sequence Step t", fontsize=11, color='#cccccc')
    ax.set_ylabel(r"Hamiltonian Energy $H(z_t) = \frac{1}{2} \|z_t\|_2^2$", fontsize=11, color='#cccccc')
    ax.set_xlim(0, L)
    ax.set_ylim(0, y_max)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='#111111', edgecolor='#444444', fontsize=8.8, loc='upper left')

    peak_idx_symp = int(np.argmax(e_symp))
    peak_idx_ph = int(np.argmax(e_ph))

    ax.annotate(
        f'Noise Trapped ($H_{{\\max}} = {symp_peak:.2f}$)',
        xy=(peak_idx_symp, e_symp[peak_idx_symp]),
        xytext=(min(peak_idx_symp + 35, 735), symp_peak * 0.92),
        arrowprops=dict(facecolor='#ff4444', shrink=0.06, width=1.2, headwidth=6),
        fontsize=9.0, color='#ff8888', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', edgecolor='#ff4444', alpha=0.85)
    )
    ax.annotate(
        f'Bounded ($H_{{\\max}} = {ph_peak:.2f}$) $\\to$ $H_T = {ph_final:.3f}$',
        xy=(peak_idx_ph, e_ph[peak_idx_ph]),
        xytext=(710, ph_peak * 1.15),
        arrowprops=dict(facecolor='#00e676', shrink=0.06, width=1.2, headwidth=6),
        fontsize=9.0, color='#00e676', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', edgecolor='#00e676', alpha=0.85)
    )

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "energy_trajectory_comparison.png")
    plt.savefig(out_path, dpi=300, facecolor='#000000', edgecolor='none')
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
