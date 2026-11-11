"""
Cartan 2x2 Port-Hamiltonian Causal Token-Mixing Operator.

Mathematical Foundations:
1. Symplectic Phase Plane: Each channel c in {1, ..., N} is a canonical 2-form pair (q_c, p_c) in R^2,
   represented as complex coordinate z_c = q_c + i * p_c in C.
2. Temporal Port-Hamiltonian Recurrence (Stage 1):
   h_{t, c} = lambda_{t, c} * h_{t-1, c} + B_{t, c} * z_{t, c}
   where lambda_{t, c} = exp(-alpha_{t, c} + i * theta_{t, c})
   - alpha_{t, c} >= 0: Gated Port-Hamiltonian Rayleigh dissipation rate (selective forgetting).
   - theta_{t, c} in R: Cartan U(1) phase angle (Hamiltonian harmonic rotation).
3. Cartan Channel Mixing (Stage 2):
   y_t = U * y^{SSM}_t
   where U in U(N) ~= Sp(2N, R) cap O(2N) via skew-Hermitian Cayley transform:
   U = (I + S)^{-1} (I - S), S^dagger = -S.
   Preserves both the canonical symplectic 2-form sum_k dq_k wedge dp_k and L2 norm identically.
4. Causality:
   d(out_t) / d(in_{t+k}) == 0 for all k > 0 identically (strictly lower-triangular).
5. Complexity:
   - Inference: O(1) state transition per token, O(N) memory cache.
   - Sequence Training: O(T) work via parallel associative prefix scan.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_associative_scan_complex(
    a: torch.Tensor,
    b: torch.Tensor,
    init_state: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Logarithmic parallel associative prefix scan for h_t = a_t * h_{t-1} + b_t.
    Binary operation: (a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2).
    Work: O(T) operations. Parallel depth: O(log T).
    
    Args:
        a: Complex tensor of shape (B, T, N) - transition coefficients.
        b: Complex tensor of shape (B, T, N) - injected inputs.
        init_state: Optional complex tensor of shape (B, N) - h_0.
    Returns:
        h: Complex tensor of shape (B, T, N) - hidden states h_1, ..., h_T.
    """
    B, T, N = a.shape
    if T == 1:
        if init_state is not None:
            return a * init_state.unsqueeze(1) + b
        return b
        
    if init_state is not None:
        # Incorporate initial state into first input
        b = b.clone()
        b[:, 0] = a[:, 0] * init_state + b[:, 0]
        # First transition effectively identity for scan formulation
        a = a.clone()
        # Initial state is now accounted for in b[:, 0]

    num_steps = (T - 1).bit_length()
    T_pad = 1 << num_steps
    
    if T_pad > T:
        a_pad = torch.ones(B, T_pad, N, dtype=a.dtype, device=a.device)
        b_pad = torch.zeros(B, T_pad, N, dtype=b.dtype, device=b.device)
        a_pad[:, :T] = a
        b_pad[:, :T] = b
    else:
        a_pad = a.clone()
        b_pad = b.clone()
        
    step = 1
    while step < T_pad:
        a_left = a_pad[:, :-step]
        b_left = b_pad[:, :-step]
        a_right = a_pad[:, step:]
        b_right = b_pad[:, step:]
        
        b_new = a_right * b_left + b_right
        a_new = a_right * a_left
        
        a_pad = torch.cat([a_pad[:, :step], a_new], dim=1)
        b_pad = torch.cat([b_pad[:, :step], b_new], dim=1)
        step *= 2
        
    return b_pad[:, :T]


class CartanChannelRotor(nn.Module):
    """
    Cartan Triple Intersection Rotor: Sp(2N, R) cap O(2N) ~= U(N).
    Parameterized via Cayley transform of skew-Hermitian generator S:
        S = 0.5 * (A - A^dagger)
        U = (I + S)^{-1} (I - S)
    Guaranteed U^dagger * U = I_N to machine precision.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # Unconstrained complex generator matrix A
        # Small initialization around zero so U starts close to identity
        scale = 1.0 / math.sqrt(n_channels)
        self.A_real = nn.Parameter(torch.randn(n_channels, n_channels) * scale)
        self.A_imag = nn.Parameter(torch.randn(n_channels, n_channels) * scale)

    def get_unitary_matrix(self) -> torch.Tensor:
        """Computes U in U(N) via Cayley transform."""
        A = torch.complex(self.A_real, self.A_imag)
        S = 0.5 * (A - A.conj().T)  # Strictly skew-Hermitian: S^H = -S
        I = torch.eye(self.n_channels, dtype=torch.complex64, device=A.device)
        # Solve (I + S) U = (I - S)
        U = torch.linalg.solve(I + S, I - S)
        return U

    def get_symplectic_orthogonal_matrix(self) -> torch.Tensor:
        """
        Returns real 2N x 2N representation M in Sp(2N, R) cap O(2N):
            M = [[X, -Y],
                 [Y,  X]]
        where U = X + iY.
        """
        U = self.get_unitary_matrix()
        X = U.real
        Y = U.imag
        top = torch.cat([X, -Y], dim=1)
        bottom = torch.cat([Y, X], dim=1)
        return torch.cat([top, bottom], dim=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, T, N) complex or (B, N) complex
        Returns U * z: (B, T, N) or (B, N)
        """
        U = self.get_unitary_matrix()
        # Matrix multiply: out = z @ U.T
        return torch.matmul(z, U.T)


class CartanCausalTokenMixer(nn.Module):
    """
    End-to-End Causal Token Mixer combining:
    1. Temporal Cartan-SSM (Stage 1):
       Parallel associative scan / O(1) sequential recurrent state-space step.
       Dissipation alpha_t >= 0 and rotation theta_t in R.
    2. Cartan U(N) Channel Rotor (Stage 2):
       Symplectic & orthogonal channel mixing via Cayley transform.
       
    I/O Shapes supported:
    - Flattened: (B, T, 2*N)
    - Factored:  (B, T, N, 2) where dim -1 is (q, p).
    """
    def __init__(
        self,
        d_model: int,
        n_channels: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        if n_channels is None:
            assert d_model % 2 == 0, f"d_model={d_model} must be even for (q, p) phase planes"
            n_channels = d_model // 2
        else:
            assert d_model == 2 * n_channels, f"d_model={d_model} must equal 2 * n_channels={2*n_channels}"
            
        self.d_model = d_model
        self.n_channels = n_channels
        
        # Input-dependent projection to SSM parameters:
        # alpha: dissipation rate (selective forgetting)
        # theta: phase rotation modulation
        # B_re, B_im: complex input injection
        self.proj_in = nn.Linear(d_model, 4 * n_channels)
        
        # Base harmonic frequencies omega_0: log-spaced across bands
        omega = torch.exp(torch.linspace(math.log(0.1), math.log(math.pi), n_channels))
        self.register_buffer("omega_0", omega)
        
        # Base log dissipation initialization (softplus around dt_min)
        self.alpha_bias = nn.Parameter(torch.full((n_channels,), -2.0))
        
        # Complex readout vector C and feedforward skip D
        self.C_real = nn.Parameter(torch.randn(n_channels) / math.sqrt(n_channels))
        self.C_imag = nn.Parameter(torch.randn(n_channels) / math.sqrt(n_channels))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Stage 2: Cartan U(N) Channel Rotor
        self.channel_rotor = CartanChannelRotor(n_channels)

    def _to_canonical_complex(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Converts input tensor into (x_flat, z_complex, is_4d_input).
        x can be (B, T, 2*N) or (B, T, N, 2).
        """
        is_4d = (x.dim() == 4 and x.shape[-1] == 2)
        if is_4d:
            B, T, N, _ = x.shape
            x_flat = x.reshape(B, T, 2 * N)
            z = torch.complex(x[..., 0], x[..., 1])
        else:
            B, T, D = x.shape
            x_flat = x
            q = x[..., :self.n_channels]
            p = x[..., self.n_channels:]
            z = torch.complex(q, p)
        return x_flat, z, is_4d

    def _from_canonical_complex(
        self,
        y_real: torch.Tensor,
        is_4d: bool
    ) -> torch.Tensor:
        """Restores output format matching input."""
        if is_4d:
            B, T, _ = y_real.shape
            q = y_real[..., :self.n_channels]
            p = y_real[..., self.n_channels:]
            return torch.stack([q, p], dim=-1)
        return y_real

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_sequential: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward causal token mixing.
        Args:
            x: (B, T, 2*N) or (B, T, N, 2)
            state: Optional initial state (B, N, 2) or complex (B, N)
            use_sequential: If True, forces sequential scan (useful for verification)
        Returns:
            out: Same shape format as x
            final_state: (B, N, 2) representing (q_h, p_h) at final step T
        """
        x_flat, z, is_4d = self._to_canonical_complex(x)
        B, T, _ = x_flat.shape
        
        # Initial complex state preparation
        init_h = None
        if state is not None:
            if state.is_complex():
                init_h = state
            elif state.dim() == 3 and state.shape[-1] == 2:
                init_h = torch.complex(state[..., 0], state[..., 1])
            elif state.dim() == 2 and state.shape[-1] == 2 * self.n_channels:
                init_h = torch.complex(state[:, :self.n_channels], state[:, self.n_channels:])
            else:
                raise ValueError(f"Unsupported state shape {state.shape}")

        # Compute gating parameters
        proj = self.proj_in(x_flat)  # (B, T, 4*N)
        p_alpha, p_theta, p_Br, p_Bi = torch.chunk(proj, 4, dim=-1)
        
        # 1. Dissipation rate alpha >= 0 (Rayleigh damping)
        alpha = F.softplus(p_alpha + self.alpha_bias)  # (B, T, N)
        
        # 2. Rotation angle theta in R (Hamiltonian rotation)
        theta = self.omega_0.unsqueeze(0).unsqueeze(0) + torch.tanh(p_theta) * math.pi
        
        # Complex transition operator lambda_t = exp(-alpha + i * theta)
        lambda_t = torch.exp(-alpha + 1j * theta)
        
        # Complex input injection B_t
        B_complex = torch.complex(p_Br, p_Bi)
        u_t = B_complex * z
        
        # Temporal recurrence across sequence dimension T
        if T == 1 or use_sequential:
            # Sequential recurrence
            h_all = torch.empty_like(u_t)
            curr = init_h if init_h is not None else torch.zeros(B, self.n_channels, dtype=torch.complex64, device=x.device)
            for t in range(T):
                curr = lambda_t[:, t] * curr + u_t[:, t]
                h_all[:, t] = curr
        else:
            # Parallel associative scan in O(T) work, O(log T) span
            h_all = parallel_associative_scan_complex(lambda_t, u_t, init_state=init_h)
            
        final_h = h_all[:, -1]
        final_state = torch.stack([final_h.real, final_h.imag], dim=-1)  # (B, N, 2)
        
        # State readout
        C = torch.complex(self.C_real, self.C_imag)
        y_ssm = C.unsqueeze(0).unsqueeze(0) * h_all  # (B, T, N)
        
        # Stage 2: Cartan U(N) Channel Rotor Mixing
        y_rot = self.channel_rotor(y_ssm)  # (B, T, N)
        
        # Recombine to real phase coordinates + skip connection
        y_real = torch.cat([y_rot.real, y_rot.imag], dim=-1)  # (B, T, 2*N)
        y_out = y_real + self.D.unsqueeze(0).unsqueeze(0) * x_flat
        
        return self._from_canonical_complex(y_out, is_4d), final_state

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) per-token autoregressive generation step.
        Args:
            x_t: (B, 2*N) or (B, N, 2) - token input at time t
            state: (B, N, 2) or complex (B, N) - state at time t-1
        Returns:
            out_t: (B, 2*N) or (B, N, 2) - token output at time t
            next_state: (B, N, 2) - updated state at time t
        """
        is_4d = (x_t.dim() == 3 and x_t.shape[-1] == 2)
        if is_4d:
            x_flat = x_t.reshape(x_t.shape[0], 2 * self.n_channels)
            z_t = torch.complex(x_t[..., 0], x_t[..., 1])
        else:
            x_flat = x_t
            q = x_t[..., :self.n_channels]
            p = x_t[..., self.n_channels:]
            z_t = torch.complex(q, p)
            
        if state.is_complex():
            h_prev = state
        elif state.dim() == 3 and state.shape[-1] == 2:
            h_prev = torch.complex(state[..., 0], state[..., 1])
        else:
            h_prev = torch.complex(state[:, :self.n_channels], state[:, self.n_channels:])
            
        # Compute gating
        proj = self.proj_in(x_flat)
        p_alpha, p_theta, p_Br, p_Bi = torch.chunk(proj, 4, dim=-1)
        
        alpha = F.softplus(p_alpha + self.alpha_bias)
        theta = self.omega_0.unsqueeze(0) + torch.tanh(p_theta) * math.pi
        lambda_t = torch.exp(-alpha + 1j * theta)
        
        B_complex = torch.complex(p_Br, p_Bi)
        u_t = B_complex * z_t
        
        # Recurrence step
        h_next = lambda_t * h_prev + u_t
        next_state = torch.stack([h_next.real, h_next.imag], dim=-1)
        
        # Readout + Cartan Rotor
        C = torch.complex(self.C_real, self.C_imag)
        y_ssm = C.unsqueeze(0) * h_next
        y_rot = torch.matmul(y_ssm, self.channel_rotor.get_unitary_matrix().T)
        
        y_real = torch.cat([y_rot.real, y_rot.imag], dim=-1) + self.D * x_flat
        
        if is_4d:
            q_out = y_real[..., :self.n_channels]
            p_out = y_real[..., self.n_channels:]
            return torch.stack([q_out, p_out], dim=-1), next_state
        return y_real, next_state
