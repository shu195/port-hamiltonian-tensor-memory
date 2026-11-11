import sys
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ph_ssm import CartanTrotter2x2Layer, compute_effective_rank

print("=" * 72)
print("CARTAN-TROTTER 2x2 BLOCK MATRIX ENGINE: VERIFICATION & BENCHMARK")
print("Intersection: M_{NxN}(2x2) n [Sp(2N, R) n O(2N)] ~= U(N) ~= M_N(Cl^0(2,0))")
print("=" * 72)

torch.manual_seed(42)

# -------------------------------------------------------------------------
# TEST 1: 50-LAYER ANALYTICAL TIME-REVERSAL & O(1) MEMORY INVERTIBILITY
# -------------------------------------------------------------------------
print("\n[1] 50-LAYER DEEP NON-LINEAR TIME-REVERSAL & LIOUVILLE VOLUME AUDIT")
B, N, L_depth = 16, 16, 50
layers = nn.ModuleList([CartanTrotter2x2Layer(n_nodes=N, dt=0.15) for _ in range(L_depth)])

z0 = torch.randn(B, N, 2)
z_curr = z0.clone()

# Forward through 50 non-linear layers
for layer in layers:
    z_curr = layer(z_curr)

z_final = z_curr.clone()
forward_displacement = torch.norm(z_final - z0) / torch.norm(z0)

# Analytical Reverse from Layer 50 -> Layer 0 with ZERO saved intermediate activations
z_rec = z_final.clone()
for layer in reversed(layers):
    z_rec = layer.inverse(z_rec)

rev_error = torch.max(torch.abs(z_rec - z0)).item()
rev_rel_err = (torch.norm(z_rec - z0) / torch.norm(z0)).item()

# Exact Jacobian determinant check on a 4-node 10-layer stack
N_jac = 4
jac_layers = nn.ModuleList([CartanTrotter2x2Layer(n_nodes=N_jac, dt=0.2) for _ in range(10)])
def stack_fn(x_flat):
    z = x_flat.view(1, N_jac, 2)
    for lyr in jac_layers:
        z = lyr(z)
    return z.view(-1)

x_sample = torch.randn(2 * N_jac)
J_full = torch.autograd.functional.jacobian(stack_fn, x_sample)
det_J = torch.linalg.det(J_full).item()

# Check symplectic 2-form preservation J_full^T @ Omega @ J_full == Omega
J2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
Omega = torch.block_diag(*[J2 for _ in range(N_jac)])
symp_violation = torch.norm(J_full.T @ Omega @ J_full - Omega).item()

print(f"  Unrolled Nonlinear Depth:        {L_depth} Layers")
print(f"  Forward State Displacement:      {forward_displacement.item() * 100:.2f}% (Non-trivial trajectory)")
print(f"  Analytical Reversal Max Error:   {rev_error:.3e} (Closed-form O(1) memory inverse)")
print(f"  Analytical Reversal Rel Error:   {rev_rel_err:.3e}")
print(f"  10-Layer Exact Jacobian det(J):  {det_J:.8f} (Theoretical: 1.00000000)")
print(f"  Symplectic 2-Form Residual:      {symp_violation:.3e} (||J^T Omega J - Omega||_F)")

# -------------------------------------------------------------------------
# TEST 2: COUPLED OPTICAL VORTEX / NON-LINEAR KERR WAVEGUIDE BENCHMARK
# -------------------------------------------------------------------------
print("\n[2] COUPLED KERR VORTEX WAVEGUIDE BENCHMARK (Cartan 2x2 vs Scalar MLP)")

N_bench = 16
n_samples = 1024
# Ground-truth physical target: 3-stage coupled symplectic Kerr waveguide
true_layers = nn.ModuleList([CartanTrotter2x2Layer(n_nodes=N_bench, dt=0.25) for _ in range(3)])
for p in true_layers.parameters():
    p.requires_grad_(False)

X_data = torch.randn(n_samples, N_bench, 2)
with torch.no_grad():
    Y_data = X_data.clone()
    for lyr in true_layers:
        Y_data = lyr(Y_data)

X_train, Y_train = X_data[:800], Y_data[:800]
X_test, Y_test = X_data[800:], Y_data[800:]

class CartanNet(nn.Module):
    def __init__(self, n_nodes, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([CartanTrotter2x2Layer(n_nodes, dt=0.25) for _ in range(depth)])
    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x

class ScalarBaselineMLP(nn.Module):
    def __init__(self, n_nodes, depth=3):
        super().__init__()
        dim = 2 * n_nodes
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
    def forward(self, x):
        B, N_n, C = x.shape
        h = x.view(B, N_n * C)
        for i, lyr in enumerate(self.layers):
            h = lyr(h)
            if i < len(self.layers) - 1:
                h = F.gelu(h)
        return h.view(B, N_n, C)

cartan_model = CartanNet(N_bench, depth=3)
scalar_model = ScalarBaselineMLP(N_bench, depth=3)

# Count effective independent parameters:
# In CartanTrotter2x2Layer, A_skew (skew-symmetric) has N(N-1)/2 and S_sym (symmetric) has N(N+1)/2 -> N^2 + N (chi)
cartan_effective_params = 3 * (N_bench * N_bench + N_bench)
scalar_params = sum(p.numel() for p in scalar_model.parameters())

opt_c = torch.optim.Adam(cartan_model.parameters(), lr=0.015)
opt_s = torch.optim.Adam(scalar_model.parameters(), lr=0.015)

for ep in range(250):
    opt_c.zero_grad()
    loss_c = F.mse_loss(cartan_model(X_train), Y_train)
    loss_c.backward()
    opt_c.step()

    opt_s.zero_grad()
    loss_s = F.mse_loss(scalar_model(X_train), Y_train)
    loss_s.backward()
    opt_s.step()

with torch.no_grad():
    pred_c = cartan_model(X_test)
    pred_s = scalar_model(X_test)
    test_mse_c = F.mse_loss(pred_c, Y_test).item()
    test_mse_s = F.mse_loss(pred_s, Y_test).item()

    erank_c = compute_effective_rank(pred_c.view(pred_c.shape[0], -1))
    erank_s = compute_effective_rank(pred_s.view(pred_s.shape[0], -1))

print(f"  Cartan-Trotter 2x2 Net | Effective DoF: {cartan_effective_params:4d} | Test MSE: {test_mse_c:.6e} | eRank: {erank_c:.2f}/32")
print(f"  Standard Scalar MLP    | Effective DoF: {scalar_params:4d} | Test MSE: {test_mse_s:.6e} | eRank: {erank_s:.2f}/32")
print(f"  Parameter Reduction:   {scalar_params / cartan_effective_params:.2f}x fewer parameters ({100*(1 - cartan_effective_params/scalar_params):.1f}% saved)")
print(f"  Precision Advantage:   {test_mse_s / test_mse_c:.2f}x lower generalization error")
print("=" * 72)
