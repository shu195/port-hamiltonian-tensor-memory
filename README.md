# Port-Hamiltonian State-Space Models (PH-SSM) for Inter-Layer Tensor Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Continuous-depth Port-Hamiltonian State-Space Model (**PH-SSM**) engineered to preserve inter-layer tensor memory across deep neural architectures without suffering from:
1. **Exponential memory amnesia** (static dissipative models where $e^{-2\alpha t} \to 0$),
2. **Brownian noise & distractor entrapment** (pure unitary / skew-symmetric models where $\operatorname{Re}(\lambda) \equiv 0$), or
3. **Dimensional rank collapse** ($\operatorname{erank}(Z) \to 1$).

Discovered and formally verified via a **Tripartite Peer-to-Peer Autonomous Agent Mesh** (Symplectic Structure $\times$ Adaptive Dissipation $\times$ Spectral Rank Audit).

---

## Mathematical Formulation

The internal state $z(t) \in \mathbb{R}^D$ evolves via the continuous-time **Port-Hamiltonian dynamical system**:

$$\frac{dz(t)}{dt} = \Big( J(x_t) - R(x_t) \Big) \nabla H(z(t)) + B(x_t) x(t)$$

where:
- **$H(z) = \frac{1}{2} \|z\|^2$** is the canonical Hamiltonian energy quadratic functional.
- **$J \in \mathfrak{so}(D)$** ($J = -J^T$) is the skew-symmetric Poisson structure tensor.
- **$R(x_t) \succeq 0$** is an adaptive, state-dependent Rayleigh dissipation operator.
- **$B(x_t)$** is a nonlinear input gating operator.

### 2nd-Order Symmetric Strang Operator Splitting
To avoid expensive online matrix inversions during inference, the flow is integrated via symmetric Strang operator splitting:

$$\Psi_{\Delta t} = \Psi_{\Delta t / 2}^R \circ \Phi_{\Delta t}^J \circ \Psi_{\Delta t / 2}^R$$

1. **First Half-Step Radial Dissipation ($\Psi_{\Delta t / 2}^R$):**
   $$z_{1/2} = \frac{z_t}{1 + \frac{\Delta t}{2} r(z_t)}, \quad r(z) = \gamma_0 \cdot \sigma\left(\frac{H(z) - H_0}{\tau}\right)$$
2. **Full-Step Cayley Symplectic Rotation ($\Phi_{\Delta t}^J$):**
   $$z^* = z_{1/2} A_J^T + \Delta t B(z_{1/2}) x_t, \quad A_J = \left(I - \frac{\Delta t}{2} J\right)^{-1} \left(I + \frac{\Delta t}{2} J\right) \in \mathrm{SO}(D)$$
3. **Second Half-Step Radial Dissipation ($\Psi_{\Delta t / 2}^R$):**
   $$z_{t+1} = \frac{z^*}{1 + \frac{\Delta t}{2} r(z^*)}$$

Because the dissipation acts **strictly radially** ($z z^T / \|z\|^2$), surplus noise energy is damped while the orthogonal $(D-1)$ tangential subspace maintains **zero dissipation**, completely eliminating dimensional rank collapse!

---

## Empirical Verification ($L=1000$ Steps, $D=32$)

Run with `python benchmark.py`:

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
└── README.md             # Formal physical derivation & verification
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
