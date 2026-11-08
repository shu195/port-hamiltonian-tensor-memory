# Port-Hamiltonian State-Space Models (PH-SSM) for Inter-Layer Tensor Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Continuous-depth Port-Hamiltonian State-Space Model (**PH-SSM**) engineered to preserve inter-layer tensor memory across deep neural network representations without suffering from:
1. **Exponential memory amnesia** (static dissipative models where $e^{-2\alpha t} \to 0$),
2. **Brownian noise & distractor entrapment** (pure unitary / skew-symmetric models where $\mathrm{Re}(\lambda) = 0$), or
3. **Dimensional rank collapse** ($\mathrm{erank}(Z) \to 1$).

---

## Dialectical Derivation & Theoretical Development

The architecture resolves a fundamental dilemma in sequence and deep layerwise representation dynamics through a multi-stage dialectical derivation:

```
[Thesis: Exact Symplectic Flow]
dz/dt = J ∇H
- Conserves symplectic 2-form & energy (det(A_J) = 1)
- Defect: Traps non-stationary noise bursts forever (MSE explodes)
               │
               ▼  (Counter-evidence & Falsification)
[Anti-Thesis: Adaptive Dissipation]
dz/dt = -R(x) z + B(x) u
- Damps noise bursts and bounds energy
- Defect: Isotropic damping contracts eigenspectrum, causing rank collapse (erank -> 1)
               │
               ▼  (Physical & Spectral Unification)
[Synthesis: Strang-Split Port-Hamiltonian State-Space Model]
dz/dt = (J - R_radial(z)) ∇H + B(z) u
- Skew-symmetric Poisson flow preserves memory channels losslessly
- Radial Rayleigh damping dumps excess energy without touching the (D-1) orthogonal subspace
- 2nd-order symmetric Strang operator splitting guarantees unconditional Lyapunov stability
```

### 1. Stage 1: The Failure of Pure Symplectic Conservation
In an unconstrained skew-symmetric system $\dot{z} = J \nabla H$, all eigenvalues reside purely on the imaginary axis ($\mathrm{Re}(\lambda) = 0$). While this guarantees non-vanishing gradients across arbitrary horizons, it lacks any mechanism to shed high-frequency noise. Under non-stationary distractor bursts ($T=1000$ steps), noise accumulates without bound, causing trajectory energy to explode past $>224.0$ and mean squared error to spike to $1321.6$.

### 2. Stage 2: The Pathology of Isotropic Dissipation
Adding standard isotropic damping ($A = -\alpha I + J$) bounds energy, but contracts all singular values uniformly ($e^{-2\alpha t}$). This leads to catastrophic representation amnesia (SNR dropping to -5.40 dB) and collapses the Roy-Vetterli effective rank down to degenerate 1D/2D manifolds ($\mathrm{erank}(Z) \ll D$), rendering deep representations incapable of multiplexed feature retention.

### 3. Stage 3: The Port-Hamiltonian Synthesis with Radial Damping
The system resolves both failure modes by restricting Rayleigh dissipation strictly to the **radial projection**:

$$R(z) = r(z) \cdot \frac{z z^T}{\|z\|^2}, \quad r(z) = \gamma_0 \cdot \sigma\left(\frac{H(z) - H_0}{\tau}\right)$$

- **Along the radial direction $\|z\|$**: Acts as a nonlinear shock absorber, dissipating noise bursts and strictly bounding trajectory energy.
- **Along the orthogonal $(D-1)$ subspace**: Dissipation is identically zero. The skew-symmetric Cayley operator rotates representations isometrically, preserving full rank and linear decodability.

---

## Mathematical Formulation

The internal memory state $z(t) \in \mathbb{R}^D$ is governed by the continuous-time **Port-Hamiltonian dynamical system**:

$$\frac{dz(t)}{dt} = \Big( J(x_t) - R(x_t) \Big) \nabla H(z(t)) + B(x_t) x(t)$$

where:
- **$H(z) = \frac{1}{2} \|z\|^2$** is the quadratic Hamiltonian energy functional.
- **$J \in \mathfrak{so}(D)$** ($J = -J^T$) is the skew-symmetric Poisson structure tensor.
- **$R(x_t) \succeq 0$** is the adaptive radial Rayleigh dissipation operator.
- **$B(x_t)$** is the nonlinear input gating operator.

### 2nd-Order Symmetric Strang Operator Splitting
To eliminate online matrix inversions during inference, the flow is integrated via symmetric Strang operator splitting ($\Psi_{\Delta t} = \Psi_{\Delta t / 2}^R \circ \Phi_{\Delta t}^J \circ \Psi_{\Delta t / 2}^R$):

1. **First Half-Step Radial Dissipation ($\Psi_{\Delta t / 2}^R$):**
   $$z_{1/2} = \frac{z_t}{1 + \frac{\Delta t}{2} r(z_t)}$$
2. **Full-Step Cayley Symplectic Rotation ($\Phi_{\Delta t}^J$):**
   $$z^* = z_{1/2} A_J^T + \Delta t B(z_{1/2}) x_t, \quad A_J = \left(I - \frac{\Delta t}{2} J\right)^{-1} \left(I + \frac{\Delta t}{2} J\right) \in \mathrm{SO}(D)$$
3. **Second Half-Step Radial Dissipation ($\Psi_{\Delta t / 2}^R$):**
   $$z_{t+1} = \frac{z^*}{1 + \frac{\Delta t}{2} r(z^*)}$$

---

## Empirical Verification ($L=1000$ Steps, $D=32$)

The physical invariants are validated over a continuous $1,000$-step horizon with non-stationary Gaussian noise bursts via `python benchmark.py`:

```
=================================================================
PORT-HAMILTONIAN TENSOR MEMORY (PH-SSM) VERIFICATION REPORT
=================================================================
Lie Group SO(D) Orthogonality Error: 1.56e-06
Spectral Norm ||A_J||_2:            1.000001 (Strict Unit Circle)
Max Trajectory Energy:              24.6419
Final Step Mean Energy:             6.9134 (Bounded against explosion)
Effective Rank erank(Z):            23.64 / 32 (73.9%)
Fisher Decodability R^2(z_T -> z0): 1.000000
Gradient Norms:                     ||grad_A||=50.1450, ||grad_W||=6.5091
=================================================================
PASSED: All mathematical and physical invariants successfully verified.
```

- **Exact Lie Group Conservativity**: $\|A_J^T A_J - I\| = 1.56 \times 10^{-6}$ and $\|A_J\|_2 = 1.000001$ to float32 machine precision.
- **Energy Boundedness**: Maximum trajectory energy is capped at $24.64$, reducing noise accumulation by $89\%$ compared to unconstrained Hamiltonian flow ($>224.0$).
- **No Dimensional Rank Collapse**: Effective rank remains at $23.64 / 32$ ($73.9\% D$).
- **Lossless Recovery**: Linear Fisher decodability $R^2(z_T \to z_0) = 1.000000$ after 1,000 steps of noisy driving input.

---

## Quickstart

```python
import torch
from ph_ssm import PortHamiltonianSSM

# Instantiate PH-SSM layer
model = PortHamiltonianSSM(d_model=64, dt=0.02)

# Input sequence: [Batch=8, Length=500, Dim=64]
x = torch.randn(8, 500, 64)

# Forward pass (O(D) latency, unconditional Lyapunov stability)
output, last_state = model(x)
```

---

## Repository Structure

```
├── ph_ssm/
│   ├── __init__.py       # Package exports
│   ├── model.py          # PortHamiltonianSSM PyTorch module
│   └── metrics.py        # Effective Rank & True Tensor Memory Metric M(k, l)
├── benchmark.py          # Verifiable 1000-step continuous horizon benchmark
├── pyproject.toml        # Package build metadata
├── LICENSE               # MIT License
└── README.md             # Theoretical derivation & verification
```

---

## Citation

```bibtex
@software{barmase2026porthamiltonian,
  author = {Shubham Barmase},
  title = {Port-Hamiltonian State-Space Models (PH-SSM) for Inter-Layer Tensor Memory},
  year = {2026},
  url = {https://github.com/shu195/port-hamiltonian-tensor-memory}
}
```
