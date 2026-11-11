import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGatedRadialRayleighDamping(nn.Module):
    """
    Token-Gated Radial Rayleigh Damping Operator for Port-Hamiltonian Sequence Models.

    Mathematical Formulation:
        R(z_t, x_t) = gamma(x_t) * (z_t z_t^T / ||z_t||^2)

    Where:
        - z_t in R^{2N} is the phase space state (or N canonical pairs (q_i, p_i)).
        - x_t is the input token representation at sequence position t.
        - gamma(x_t) in [0, gamma_max] is an input-dependent dissipation scalar:
            gamma(x_t) = gamma_max * sigmoid(W_gamma @ x_t + b_gamma)
          Low-information filler tokens ("the", "a", commas): gamma(x_t) -> gamma_max.
          Dense semantic keyword tokens (nouns, verbs): gamma(x_t) -> 0.

    Algebraic and Geometric Invariants:
    1. Spectral Decomposition:
       The operator R(z, x) is a rank-1 projection onto the radial line span(z):
           R(z, x) = gamma(x) * P_parallel(z)
           where P_parallel(z) = z z^T / ||z||^2.
       Its eigenvalues are:
           lambda_1 = gamma(x) >= 0 (1-dimensional eigenspace V_parallel = span(z))
           lambda_2 = ... = lambda_{2N} = 0 (2N-1 dimensional eigenspace V_perp = {v | z^T v = 0})

    2. Energy Purging along the Radial Ray:
       Under the Hamiltonian carrier H(z) = 0.5 * ||z||^2, the continuous damping force is:
           dz/dt = -R(z, x) grad H(z) = -gamma(x) * z
           dH/dt = -2 * gamma(x) * H
       Yielding exact exponential dissipation of filler energy:
           ||z(t)||^2 = ||z(0)||^2 * exp(-2 * int_0^t gamma(x(s)) ds)

    3. Isometric Invariance along the (2N-1) Orthogonal Subspace:
       For any vector v_perp in V_perp (z^T v_perp = 0):
           R(z, x) v_perp = 0
       And the matrix exponential flow operator is:
           exp(-dt * R(z, x)) = (I - P_parallel(z)) + exp(-gamma(x) * dt) * P_parallel(z)
           exp(-dt * R(z, x)) v_perp = v_perp identically!
       The orthogonal spectrum maintains eigenvalue lambda == 1.0000000 with zero drift,
       preventing anisotropic rank collapse.

    4. Directional State Preservation:
       Under combined Port-Hamiltonian flow dz/dt = (J(x) - R(z, x)) z:
           d/dt (z / ||z||) = J(x) (z / ||z||)
       The damping term completely drops out of the spherical dynamics, rotating
       the directional semantic phase isometrically on S^{2N-1}.
    """

    def __init__(
        self,
        d_model: int,
        in_dim: Optional[int] = None,
        dt: float = 0.1,
        gamma_max: float = 2.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_dim = in_dim if in_dim is not None else d_model
        self.dt = dt
        self.gamma_max = gamma_max
        self.eps = eps

        # Token gating scalar projection: maps token representation x to scalar gamma
        self.gate_proj = nn.Linear(self.in_dim, 1)
        self.gate_scale = nn.Parameter(torch.tensor(float(gamma_max)))

        # Initialize gate projection with small weights
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.zeros_(self.gate_proj.bias)

    def compute_gamma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the dissipation scalar gamma(x) in [0, gamma_max].
        Args:
            x: Input tensor [..., in_dim]
        Returns:
            gamma: Gating scalar [..., 1]
        """
        g_raw = torch.sigmoid(self.gate_proj(x))
        return torch.clamp(self.gate_scale, min=0.0) * g_raw

    def forward(
        self,
        z: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies exact Token-Gated Radial Rayleigh Damping to state z:
            z' = exp(-gamma(x) * dt) * z

        Args:
            z: State tensor of shape [..., 2*N] or [..., N, 2]
            x: Optional driving token representation [..., in_dim]
            gamma: Optional pre-computed scalar gating [..., 1]
            dt: Optional time step override (defaults to self.dt)

        Returns:
            z_damped: Purged state matching input shape
            gamma: Gating scalar applied [..., 1]
        """
        if gamma is None:
            if x is None:
                raise ValueError("Either 'x' or 'gamma' must be provided.")
            gamma = self.compute_gamma(x)

        step_dt = self.dt if dt is None else dt

        # Support both (..., N, 2) and (..., 2*N) representations
        orig_shape = z.shape
        if orig_shape[-1] == 2 and len(orig_shape) >= 2:
            # Shape is (..., N, 2)
            # Radial contraction factor: exp(-gamma * dt)
            # Ensure gamma broadcasts across N and 2
            decay = torch.exp(-gamma * step_dt)
            while decay.dim() < z.dim():
                decay = decay.unsqueeze(-1)
            z_damped = z * decay
        else:
            decay = torch.exp(-gamma * step_dt)
            while decay.dim() < z.dim():
                decay = decay.unsqueeze(-1)
            z_damped = z * decay

        return z_damped, gamma

    def apply_operator(
        self,
        z_ref: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Applies the exact full matrix exponential flow operator exp(-dt * R(z_ref)) to an arbitrary vector v:
            exp(-dt * R(z_ref)) v = v - (1.0 - exp(-gamma * dt)) * (<z_ref, v> / ||z_ref||^2) * z_ref

        Invariants:
            - If v is parallel to z_ref (v = c * z_ref): exp(-dt * R) v = exp(-gamma * dt) * v
            - If v is orthogonal to z_ref (<z_ref, v> = 0): exp(-dt * R) v = v (identically)
            - Asymptotic complexity: O(D) operations with ZERO matrix inversion or O(D^2) dense GEMM.
        """
        step_dt = self.dt if dt is None else dt
        # Flatten last dimensions if necessary
        z_flat = z_ref.reshape(*z_ref.shape[:-2], -1) if z_ref.shape[-1] == 2 else z_ref
        v_flat = v.reshape(*v.shape[:-2], -1) if v.shape[-1] == 2 else v

        # Compute norm squared of z_ref
        z_norm_sq = torch.sum(z_flat * z_flat, dim=-1, keepdim=True) + self.eps
        # Compute projection inner product <z_ref, v>
        inner_prod = torch.sum(z_flat * v_flat, dim=-1, keepdim=True)

        decay = torch.exp(-gamma * step_dt)
        # Closed form: v - (1 - decay) * (<z, v> / ||z||^2) * z
        v_out_flat = v_flat - (1.0 - decay) * (inner_prod / z_norm_sq) * z_flat
        return v_out_flat.reshape_as(v)

    def explicit_dissipation_matrix(
        self,
        z: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Explicitly constructs the 2N x 2N rank-1 dissipation matrix:
            R = gamma * (z z^T / ||z||^2)
        Used for Jacobian, spectral, and Lie algebra verification.
        """
        z_flat = z.reshape(-1)
        norm_sq = torch.dot(z_flat, z_flat) + self.eps
        P_parallel = torch.outer(z_flat, z_flat) / norm_sq
        g = gamma.reshape(-1)[0]
        return g * P_parallel
