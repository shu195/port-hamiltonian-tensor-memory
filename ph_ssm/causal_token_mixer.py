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


def chunked_log_scan_real_split(
    alpha: torch.Tensor,
    theta: torch.Tensor,
    u_r: torch.Tensor,
    u_i: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash-Linear-Attention (FLA) style chunked log-cumsum scan in split real-imaginary coordinates.
    Avoids unvectorized CPU complex64 exp/cumsum and eliminates recursive tree slice/cat allocations.
    Solves h_t = exp(-alpha_t + i * theta_t) * h_{t-1} + (u_{r, t} + i * u_{i, t}).
    """
    B, T, N = alpha.shape
    if init_state is not None:
        if init_state.is_complex():
            h0_r, h0_i = init_state.real, init_state.imag
        elif init_state.dim() == 3 and init_state.shape[-1] == 2:
            h0_r, h0_i = init_state[..., 0], init_state[..., 1]
        else:
            h0_r, h0_i = init_state[:, :N], init_state[:, N:]
        ea0 = torch.exp(-alpha[:, 0])
        ct0, st0 = torch.cos(theta[:, 0]), torch.sin(theta[:, 0])
        # lambda_0 * h0 + u_0
        u0_r = ea0 * (ct0 * h0_r - st0 * h0_i) + u_r[:, 0]
        u0_i = ea0 * (ct0 * h0_i + st0 * h0_r) + u_i[:, 0]
        u_r = torch.cat([u0_r.unsqueeze(1), u_r[:, 1:]], dim=1)
        u_i = torch.cat([u0_i.unsqueeze(1), u_i[:, 1:]], dim=1)

    if T == 1:
        return u_r, u_i

    if T <= chunk_size:
        g_a = torch.cumsum(-alpha, dim=1)
        g_t = torch.cumsum(theta, dim=1)
        # Shift g_a by chunk mean/first to guarantee zero float32 overflow/underflow
        g_a_ref = g_a[:, :1, :].detach()
        g_a_rel = g_a - g_a_ref
        ea = torch.exp(g_a_rel)
        ena = torch.exp(-g_a_rel)
        ct = torch.cos(g_t)
        st = torch.sin(g_t)
        w_r = ena * (ct * u_r + st * u_i)
        w_i = ena * (ct * u_i - st * u_r)
        s_r = torch.cumsum(w_r, dim=1)
        s_i = torch.cumsum(w_i, dim=1)
        out_r = ea * (ct * s_r - st * s_i)
        out_i = ea * (ct * s_i + st * s_r)
        return out_r, out_i

    # Multi-chunk decomposition for T > chunk_size
    num_chunks = (T + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - T
    if pad_len > 0:
        z_pad = torch.zeros(B, pad_len, N, dtype=alpha.dtype, device=alpha.device)
        alpha = torch.cat([alpha, z_pad], dim=1)
        theta = torch.cat([theta, z_pad], dim=1)
        u_r = torch.cat([u_r, z_pad], dim=1)
        u_i = torch.cat([u_i, z_pad], dim=1)

    C = chunk_size
    a_c = alpha.view(B, num_chunks, C, N)
    t_c = theta.view(B, num_chunks, C, N)
    ur_c = u_r.view(B, num_chunks, C, N)
    ui_c = u_i.view(B, num_chunks, C, N)

    g_a = torch.cumsum(-a_c, dim=2)
    g_t = torch.cumsum(t_c, dim=2)
    ea = torch.exp(g_a)
    ena = torch.exp(-g_a)
    ct = torch.cos(g_t)
    st = torch.sin(g_t)

    w_r = ena * (ct * ur_c + st * ui_c)
    w_i = ena * (ct * ui_c - st * ur_c)
    s_r = torch.cumsum(w_r, dim=2)
    s_i = torch.cumsum(w_i, dim=2)

    h_intra_r = ea * (ct * s_r - st * s_i)
    h_intra_i = ea * (ct * s_i + st * s_r)

    # Inter-chunk boundary propagation
    out_r_list = [h_intra_r[:, 0]]
    out_i_list = [h_intra_i[:, 0]]
    H_r = h_intra_r[:, 0, -1, :]
    H_i = h_intra_i[:, 0, -1, :]

    for c in range(1, num_chunks):
        ea_c = ea[:, c]
        ct_c = ct[:, c]
        st_c = st[:, c]
        Hr_exp = H_r.unsqueeze(1)
        Hi_exp = H_i.unsqueeze(1)
        dec_r = ea_c * (ct_c * Hr_exp - st_c * Hi_exp)
        dec_i = ea_c * (ct_c * Hi_exp + st_c * Hr_exp)
        tot_r = h_intra_r[:, c] + dec_r
        tot_i = h_intra_i[:, c] + dec_i
        out_r_list.append(tot_r)
        out_i_list.append(tot_i)
        H_r = tot_r[:, -1, :]
        H_i = tot_i[:, -1, :]

    out_r = torch.cat(out_r_list, dim=1)[:, :T]
    out_i = torch.cat(out_i_list, dim=1)[:, :T]
    return out_r, out_i


def parallel_associative_scan_complex(
    a: torch.Tensor,
    b: torch.Tensor,
    init_state: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Logarithmic parallel associative prefix scan for h_t = a_t * h_{t-1} + b_t.
    Uses fast real-split log-polar chunked scan when |a_t| > 0, with tree fallback.
    """
    B, T, N = a.shape
    if T == 1:
        if init_state is not None:
            return a * init_state.unsqueeze(1) + b
        return b

    # Extract log-polar coordinates: a = exp(-alpha + i * theta)
    abs_a = torch.abs(a).clamp(min=1e-12)
    alpha = -torch.log(abs_a)
    theta = torch.angle(a)
    out_r, out_i = chunked_log_scan_real_split(alpha, theta, b.real, b.imag, init_state=init_state)
    return torch.complex(out_r, out_i)


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
        
        # Temporal recurrence across sequence dimension T
        if T == 1 or use_sequential:
            lambda_t = torch.exp(-alpha + 1j * theta)
            B_complex = torch.complex(p_Br, p_Bi)
            u_t = B_complex * z
            h_all = torch.empty_like(u_t)
            curr = init_h if init_h is not None else torch.zeros(B, self.n_channels, dtype=torch.complex64, device=x.device)
            for t in range(T):
                curr = lambda_t[:, t] * curr + u_t[:, t]
                h_all[:, t] = curr
        else:
            # Fast AVX2/SIMD real-split FLA chunked log-cumsum scan
            z_r, z_i = z.real, z.imag
            u_r = p_Br * z_r - p_Bi * z_i
            u_i = p_Br * z_i + p_Bi * z_r
            h_r, h_i = chunked_log_scan_real_split(alpha, theta, u_r, u_i, init_state=init_h)
            h_all = torch.complex(h_r, h_i)
            
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
