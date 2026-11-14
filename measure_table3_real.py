import math
import torch
import torch.nn as nn
from ph_ssm import PortHamiltonianSSM, compute_effective_rank, compute_decodability_r2

torch.manual_seed(42)
B, L, D = 128, 1000, 32
dt = 0.02

# Signal + distractor burst on [200, 700) matching gen_real_energy.py
t = torch.linspace(0, 20, L)
clean_signal = torch.sin(0.5 * t).unsqueeze(0).unsqueeze(-1).repeat(B, 1, D)
u = clean_signal + 0.1 * torch.randn(B, L, D)
u[:, 200:700, :] += 10.0 * torch.randn(B, 500, D)
z0 = torch.randn(B, D) * math.sqrt(2.0 * 1.0 / D)

def eval_model(name, model, iso_gamma=None):
    with torch.no_grad():
        if iso_gamma is None:
            _, states = model(u, z0=z0, return_states=True)
        else:
            A_J = model.get_cayley_operator()
            z = z0.clone()
            decay = math.exp(-iso_gamma * dt)
            traj = []
            for k in range(L):
                gate = torch.sigmoid(torch.matmul(z, model.W_gate) + model.b_gate)
                u_gated = gate * u[:, k, :]
                z = decay * (torch.matmul(z, A_J.T) + dt * u_gated)
                traj.append(z)
            states = torch.stack(traj, dim=1)

    energies = 0.5 * torch.sum(states ** 2, dim=-1)
    peak_e = energies.mean(dim=0).max().item()
    final_e = energies[:, -1].mean().item()
    z_T = states[:, -1, :]

    # Linear ridge reconstruction of z0 from z_T
    lam = 1e-4
    W_star = torch.linalg.solve(z_T.T @ z_T + lam * torch.eye(D), z_T.T @ z0)
    z0_hat = z_T @ W_star
    mse = torch.mean((z0_hat - z0) ** 2).item()
    sig_pow = torch.mean(z0 ** 2).item()
    snr = 10.0 * math.log10(sig_pow / max(mse, 1e-12))

    erank = compute_effective_rank(states.reshape(-1, D))
    r2_val = compute_decodability_r2(z_T, states[:, 0, :])
    er0 = compute_effective_rank(states[:, 0, :])
    erT = compute_effective_rank(z_T)
    diffs = torch.norm(states[:, 1:, :] - states[:, :-1, :], dim=-1).mean(dim=0)
    norms = torch.norm(states[:, :-1, :], dim=-1).mean(dim=0).clamp(min=1e-8)
    work = torch.sum(diffs / norms).item()
    m_val = max(0.0, r2_val) * (min(er0, erT) / D) * (1.0 - math.exp(-work))
    print(f"{name:22s} | MSE={mse:.4f} | SNR={snr:+.2f} dB | Final={final_e:.4f} (Peak={peak_e:.2f}) | eRank={erank:.2f}/32 ({erank/32*100:.1f}%) | M={m_val:.3f}")

m_symp = PortHamiltonianSSM(d_model=D, dt=dt, gamma_0=0.0)
m_ph = PortHamiltonianSSM(d_model=D, dt=dt, gamma_0=0.15, h_target=1.0, tau=0.5)
# Copy weights so comparison is strictly across damping regimes
m_ph.load_state_dict(m_symp.state_dict())

eval_model("Pure Symplectic (R=0)", m_symp)
eval_model("Isotropic Damped", m_symp, iso_gamma=0.5)
eval_model("PH-SSM (Radial R)", m_ph)

# 10,000-step BPTT Jacobian Frobenius norm for unforced A_J^T
A_J = m_ph.get_cayley_operator().detach()
z_init = torch.randn(D, requires_grad=True)
z_curr = z_init
for _ in range(10000):
    z_curr = A_J @ z_curr
# Compute Jacobian Frobenius norm via power/autograd trace over D basis vectors
J_mat = []
for i in range(D):
    g = torch.autograd.grad(z_curr[i], z_init, retain_graph=True)[0]
    J_mat.append(g)
J_mat = torch.stack(J_mat, dim=0)
frob_10k = torch.norm(J_mat, p='fro').item()
drift_pct = abs(frob_10k - math.sqrt(D)) / math.sqrt(D) * 100.0
print(f"10,000-step ||d z_T / d z_0||_F = {frob_10k:.6f} (sqrt(32)={math.sqrt(D):.6f}, drift={drift_pct:.5f}%)")
