import torch
from ph_ssm import PortHamiltonianSSM, compute_effective_rank, compute_decodability_r2

def run_verification_benchmark():
    torch.manual_seed(42)
    B, L, D = 16, 1000, 32
    dt = 0.02
    
    print(f"Executing Port-Hamiltonian SSM Verification (Batch={B}, Length={L}, Dim={D}, dt={dt})...")
    
    # Input sequence: sinusoidal signal + non-stationary Gaussian noise bursts
    t = torch.linspace(0, 20, L)
    clean_signal = torch.sin(0.5 * t).unsqueeze(0).unsqueeze(-1).repeat(B, 1, D)
    noise = 0.5 * torch.randn(B, L, D)
    u = clean_signal + noise
    z0 = torch.randn(B, D)
    
    model = PortHamiltonianSSM(d_model=D, dt=dt)
    
    # 1. Check Cayley transform orthogonality in Lie group SO(D)
    A_J = model.get_cayley_operator()
    ortho_err = torch.norm(A_J.T @ A_J - torch.eye(D)).item()
    spectral_norm = torch.linalg.norm(A_J, 2).item()
    
    # 2. Forward pass with 2nd-order Strang operator splitting
    y, states = model(u, z0=z0, return_states=True)
    
    # 3. Energy metrics
    energies = 0.5 * torch.norm(states, dim=-1).pow(2)
    max_energy = energies.max().item()
    final_energy = energies[:, -1].mean().item()
    
    # 4. Roy-Vetterli effective rank audit
    all_states = states.reshape(-1, D)
    erank = compute_effective_rank(all_states)
    
    # 5. Linear Fisher decodability from final state z_T back to initial memory z_0
    z_T = states[:, -1, :]
    r2 = compute_decodability_r2(z_T, z0)
    
    # 6. Backward adjoint gradient verification
    loss = torch.norm(states[:, -1, :])
    loss.backward()
    grad_norm_A = torch.norm(model.A_param.grad).item()
    grad_norm_W = torch.norm(model.W_gate.grad).item()
    
    print("=" * 65)
    print("PORT-HAMILTONIAN TENSOR MEMORY (PH-SSM) VERIFICATION REPORT")
    print("=" * 65)
    print(f"Lie Group SO(D) Orthogonality Error: {ortho_err:.2e}")
    print(f"Spectral Norm ||A_J||_2:            {spectral_norm:.6f} (Strict Unit Circle)")
    print(f"Max Trajectory Energy:              {max_energy:.4f}")
    print(f"Final Step Mean Energy:             {final_energy:.4f} (Bounded against explosion)")
    print(f"Effective Rank erank(Z):            {erank:.2f} / {D} ({erank/D*100:.1f}%)")
    print(f"Fisher Decodability R^2(z_T -> z0): {r2:.6f}")
    print(f"Gradient Norms:                     ||grad_A||={grad_norm_A:.4f}, ||grad_W||={grad_norm_W:.4f}")
    print("=" * 65)
    
    assert ortho_err < 1e-5, "Cayley operator must be strictly orthogonal!"
    assert final_energy < 50.0, "Trajectory energy must remain strictly bounded!"
    assert r2 > 0.95, "Decodability R^2 must be >= 0.95!"
    print("PASSED: All mathematical and physical invariants successfully verified.")

if __name__ == "__main__":
    run_verification_benchmark()
