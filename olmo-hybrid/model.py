"""
OLMo Hybrid 7B — MLX implementation (mlx-lm compatible)

Architecture:
  • 32 layers in 3:1 pattern (3× GatedDeltaNet, 1× full Attention), repeated 8×
  • GatedDeltaNet: delta-rule recurrence + short causal conv on q/k/v
  • Attention: causal MHA with RoPE and QK-norm (OLMo2 post-norm style)

mlx-lm usage:
    mlx_lm.generate --model allenai/OLMo-Hybrid-7B --prompt "..."

Reference: allenai/OLMo-Hybrid-7B (HuggingFace transformers 5.9+)
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Config  (field names match config.json for mlx-lm auto-loading)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArgs:
    model_type: str = "olmo_hybrid"

    # Standard transformer
    vocab_size: int = 100352
    hidden_size: int = 3840
    num_hidden_layers: int = 32
    num_attention_heads: int = 30
    num_key_value_heads: int = 30
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0        # OLMo2 default

    # Layer-type schedule (list of "linear_attention" | "full_attention")
    layer_types: List[str] = field(default_factory=list)

    # GatedDeltaNet (linear_attention) parameters
    linear_num_key_heads: int = 30
    linear_num_value_heads: int = 30
    linear_key_head_dim: int = 96       # D_k
    linear_value_head_dim: int = 192    # D_v
    linear_conv_kernel_dim: int = 4     # ShortConv kernel size K
    linear_allow_neg_eigval: bool = True

    def __post_init__(self):
        if not self.layer_types:
            # Fallback: infer 3:1 pattern (shouldn't be needed with config.json present)
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_kv_heads(self) -> int:  # alias expected by mlx-lm utilities
        return self.num_key_value_heads


# ─────────────────────────────────────────────────────────────────────────────
# Cache objects
# ─────────────────────────────────────────────────────────────────────────────

class GDNCache:
    """
    Per-layer cache for GatedDeltaNet.

    State: (S, conv_ctx)
      S:        mx.array (B, H, D_v, D_k)  — associative memory matrix
      conv_ctx: (q_ctx, k_ctx, v_ctx)       — last K-1 raw projected tokens
                each mx.array (B, K-1, H*dim)
    """

    def __init__(self):
        self._state: Optional[Tuple] = None

    @property
    def state(self) -> Optional[Tuple]:
        return self._state

    @state.setter
    def state(self, v: Optional[Tuple]):
        self._state = v

    @property
    def offset(self) -> int:
        return 0   # position tracked via recurrence, not explicit offset


class KVCache:
    """
    Standard KV cache for full-attention layers.
    Grows in steps of `step` tokens to amortise re-allocation.
    """

    def __init__(self, head_dim: int, n_heads: int, step: int = 256):
        self.head_dim = head_dim
        self.n_heads  = n_heads
        self.step     = step
        self.keys:   Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update_and_fetch(
        self,
        keys:   mx.array,   # (B, H, T, D)
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        B, H, T, D = keys.shape
        prev       = self._offset
        new_offset = prev + T

        if self.keys is None or new_offset > self.keys.shape[2]:
            n_steps  = math.ceil(new_offset / self.step)
            capacity = n_steps * self.step
            new_k = mx.zeros((B, H, capacity, D), dtype=keys.dtype)
            new_v = mx.zeros((B, H, capacity, D), dtype=values.dtype)
            if self.keys is not None:
                new_k = mx.concatenate([self.keys[:, :, :prev, :],
                                        mx.zeros((B, H, capacity - prev, D), dtype=keys.dtype)], axis=2)
                new_v = mx.concatenate([self.values[:, :, :prev, :],
                                        mx.zeros((B, H, capacity - prev, D), dtype=values.dtype)], axis=2)
            self.keys   = new_k
            self.values = new_v

        self.keys   = mx.concatenate([self.keys[:, :, :prev, :], keys,
                                      self.keys[:, :, new_offset:, :]], axis=2)
        self.values = mx.concatenate([self.values[:, :, :prev, :], values,
                                      self.values[:, :, new_offset:, :]], axis=2)
        self._offset = new_offset
        return self.keys[:, :, :new_offset, :], self.values[:, :, :new_offset, :]


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps    = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ShortConv(nn.Module):
    """
    Causal depthwise conv1d + SiLU.

    Uses mx.conv1d with groups=C for native MLX depthwise support.

    Weight shape: (C, K, 1)  [MLX conv1d: (out_ch, kW, in_ch // groups)]
    HF checkpoint stores (C, 1, K) — sanitize() transposes to (C, K, 1).

    The `prefix` argument passes real prior-token context during decode,
    avoiding the zero-padding bug that corrupts all tokens after the first.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels    = channels
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, kernel_size, 1))  # (C, K, 1)

    def __call__(
        self,
        x:      mx.array,
        prefix: Optional[mx.array] = None,
    ) -> mx.array:
        """
        x:      (B, T, C)
        prefix: (B, K-1, C) real prior context, or None → zero-pad (prefill)
        Returns (B, T, C)
        """
        B, T, C = x.shape
        K   = self.kernel_size
        pad = prefix if prefix is not None else mx.zeros((B, K - 1, C))
        x_pad = mx.concatenate([pad, x], axis=1)                    # (B, K-1+T, C)
        out   = mx.conv1d(x_pad, self.weight, padding=0, groups=C)  # (B, T, C)
        return nn.silu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel scan for GDN
# ─────────────────────────────────────────────────────────────────────────────

def _gdn_parallel_scan(
    A_all:  mx.array,   # (B, H, T, Dk, Dk)
    b_all:  mx.array,   # (B, H, T, Dv, Dk)
    S_init: mx.array,   # (B, H, Dv, Dk)
) -> mx.array:
    """
    Parallel prefix scan for the GDN linear recurrence S_t = S_{t-1} @ A_t + b_t.

    Associative operator:  (A1, b1) ⊕ (A2, b2)  =  (A1 @ A2,  b1 @ A2 + b2)

    Hillis-Steele inclusive scan — O(log T) serial depth, O(T log T) work.
    Each level is a single batched matmul over the full sequence, so all T
    positions are updated in parallel on the GPU.

    After the scan, A_scan[:,:,t] and b_scan[:,:,t] encode the cumulative
    transformation from step 0 to step t:
        S_t  =  S_init @ A_scan[:,:,t]  +  b_scan[:,:,t]

    Returns S_all: (B, H, T, Dv, Dk)  — state after each of the T tokens.
    """
    T = A_all.shape[2]

    # Upcast to float32 — composing log₂(T) levels of float16 matmuls
    # accumulates rounding error that can overflow to NaN at T=256.
    # Cast back to original dtype on exit so the rest of the model is unaffected.
    orig_dtype = A_all.dtype
    A_scan = A_all.astype(mx.float32)
    b_scan = b_all.astype(mx.float32)
    S_f32  = S_init.astype(mx.float32)

    stride = 1
    while stride < T:
        n = T - stride
        # Left operand: positions 0 .. T-stride-1
        # Right operand: positions stride .. T-1
        # Combined: left ⊕ right  →  stored at right positions
        new_right_A = A_scan[:, :, :n] @ A_scan[:, :, stride:]
        new_right_b = b_scan[:, :, :n] @ A_scan[:, :, stride:] + b_scan[:, :, stride:]
        A_scan = mx.concatenate([A_scan[:, :, :stride], new_right_A], axis=2)
        b_scan = mx.concatenate([b_scan[:, :, :stride], new_right_b], axis=2)
        stride *= 2

    # S_all[t] = S_init @ A_scan[t] + b_scan[t]
    # S_init[:,:,None] broadcasts (B,H,1,Dv,Dk) over T
    result = S_f32[:, :, None] @ A_scan + b_scan   # (B, H, T, Dv, Dk)
    return result.astype(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# GatedDeltaNet  (linear_attention layers)
# ─────────────────────────────────────────────────────────────────────────────

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet recurrent layer.

    Per-token recurrence (per head h):
        dt    = softplus(a_proj(x) + dt_bias)           — always > 0
        alpha = exp(dt · (−exp(A_log)))  ∈ (0, 1]       — per-head decay
        beta  = sigmoid(b_proj(x)) · 2  ∈ (0, 2]        — write scale
                (×2 from allow_neg_eigval: allows beta > 1 → negative
                 eigenvalues in the write-op (I − β k kᵀ), NOT in the decay)

        k  = L2_norm(ShortConv(k_proj(x)))
        q  = L2_norm(ShortConv(q_proj(x))) / √D_k      — 1/√D_k matches HF kernel
        v  = ShortConv(v_proj(x))

        S ← alpha·S + beta·(v − S@k) ⊗ k              — delta-rule state update
        y ← o_norm(S @ q) ⊙ silu(g_proj(x))           — gated readout
        out ← o_proj(y)

    Note: chunked/parallel-scan recurrence would speed up long-context prefill;
    that's a natural follow-up once the sequential baseline is merged.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        H   = args.linear_num_key_heads
        D_k = args.linear_key_head_dim
        D_v = args.linear_value_head_dim
        hid = args.hidden_size

        self.num_heads        = H
        self.key_dim          = D_k
        self.val_dim          = D_v
        self.allow_neg_eigval = args.linear_allow_neg_eigval
        self.q_scale          = 1.0 / math.sqrt(D_k)

        self.q_proj = nn.Linear(hid, H * D_k, bias=False)
        self.k_proj = nn.Linear(hid, H * D_k, bias=False)
        self.v_proj = nn.Linear(hid, H * D_v, bias=False)
        self.b_proj = nn.Linear(hid, H,       bias=False)   # beta
        self.g_proj = nn.Linear(hid, H * D_v, bias=False)   # output gate
        self.a_proj = nn.Linear(hid, H,       bias=False)   # dt input

        # Mamba-style decay parameterisation
        self.dt_bias = mx.zeros((H,))   # learned bias added to dt
        self.A_log   = mx.zeros((H,))   # stores log(|A|); decay A = −exp(A_log)

        K = args.linear_conv_kernel_dim
        self.q_conv1d = ShortConv(H * D_k, kernel_size=K)
        self.k_conv1d = ShortConv(H * D_k, kernel_size=K)
        self.v_conv1d = ShortConv(H * D_v, kernel_size=K)

        self.o_norm = RMSNorm(D_v, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(H * D_v, hid, bias=False)

    @staticmethod
    def _l2_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + eps)

    def __call__(
        self,
        x:     mx.array,
        cache: Optional[GDNCache] = None,
    ) -> mx.array:
        """
        x:     (B, T, hidden)
        cache: GDNCache updated in-place; pass None for stateless / training use
        """
        B, T, _ = x.shape
        H, D_k, D_v = self.num_heads, self.key_dim, self.val_dim
        pad_len = self.q_conv1d.kernel_size - 1   # = K-1 = 3

        # ── Unpack cache ──────────────────────────────────────────────────────
        if cache is None or cache.state is None:
            S        = mx.zeros((B, H, D_v, D_k))
            conv_ctx = None
        else:
            S, conv_ctx = cache.state

        q_ctx = k_ctx = v_ctx = None
        if conv_ctx is not None:
            q_ctx, k_ctx, v_ctx = conv_ctx

        # ── Projections ───────────────────────────────────────────────────────
        q_raw = self.q_proj(x)   # (B, T, H*D_k)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)   # (B, T, H*D_v)

        beta  = mx.sigmoid(self.b_proj(x))                    # (B, T, H)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        gate  = nn.silu(self.g_proj(x))                       # (B, T, H*D_v)

        dt    = nn.softplus(self.a_proj(x) + self.dt_bias)    # (B, T, H)
        A     = -mx.exp(self.A_log)                           # (H,) always < 0
        alpha = mx.exp(dt * A)                                # (B, T, H) ∈ (0,1]

        # ── ShortConv with causal context ─────────────────────────────────────
        # For prefill (conv_ctx=None): zero-pad — correct at sequence start.
        # For decode:  conv_ctx holds last K-1 real tokens — no zero-pad bug.
        q = self.q_conv1d(q_raw, prefix=q_ctx)   # (B, T, H*D_k)
        k = self.k_conv1d(k_raw, prefix=k_ctx)
        v = self.v_conv1d(v_raw, prefix=v_ctx)   # (B, T, H*D_v)

        # Update conv context: save last pad_len raw (pre-conv) tokens
        if T >= pad_len:
            new_conv_ctx = (
                q_raw[:, -pad_len:],
                k_raw[:, -pad_len:],
                v_raw[:, -pad_len:],
            )
        else:
            # Short input (T < K-1): slide existing context forward
            def _slide(old, new, C):
                base = old if old is not None else mx.zeros((B, pad_len, C))
                return mx.concatenate([base, new], axis=1)[:, -pad_len:]
            new_conv_ctx = (
                _slide(q_ctx, q_raw, q_raw.shape[-1]),
                _slide(k_ctx, k_raw, k_raw.shape[-1]),
                _slide(v_ctx, v_raw, v_raw.shape[-1]),
            )

        # ── Per-head reshape, L2-norm, 1/√D_k scale ──────────────────────────
        q = self._l2_norm(q.reshape(B, T, H, D_k)) * self.q_scale   # (B,T,H,D_k)
        k = self._l2_norm(k.reshape(B, T, H, D_k))
        v = v.reshape(B, T, H, D_v)

        alpha = alpha.reshape(B, T, H, 1, 1)   # broadcastable over (D_v, D_k)
        beta  = beta.reshape(B, T, H, 1, 1)
        gate  = gate.reshape(B, T, H, D_v)

        # ── Parallel associative scan (O(log T) depth vs O(T) sequential) ───────
        # S_t = S_{t-1} @ A_t + b_t  where
        #   A_t = alpha_t * I  -  beta_t * k_t ⊗ k_t^T   (Dk, Dk)
        #   b_t = beta_t * v_t ⊗ k_t                      (Dv, Dk)
        # Associative operator: (A1,b1) ⊕ (A2,b2) = (A1@A2, b1@A2+b2)
        I_k  = mx.eye(D_k, dtype=x.dtype)                            # (Dk, Dk)
        kk   = k[..., :, None] * k[..., None, :]                     # (B,T,H,Dk,Dk)
        A_all = (alpha * I_k - beta * kk).transpose(0, 2, 1, 3, 4)  # (B,H,T,Dk,Dk)

        vk   = v[..., :, None] * k[..., None, :]                     # (B,T,H,Dv,Dk)
        b_all = (beta * vk).transpose(0, 2, 1, 3, 4)                 # (B,H,T,Dv,Dk)

        S_all = _gdn_parallel_scan(A_all, b_all, S)                  # (B,H,T,Dv,Dk)

        # y_t = S_t @ q_t
        q_bhTk = q.transpose(0, 2, 1, 3)                             # (B,H,T,Dk)
        y = (S_all @ q_bhTk[..., None]).squeeze(-1)                  # (B,H,T,Dv)
        y = y.transpose(0, 2, 1, 3)                                  # (B,T,H,Dv)

        S = S_all[:, :, -1]    # final recurrent state for cache: (B,H,Dv,Dk)

        # ── Output: RMSNorm → gate → project ─────────────────────────────────
        y = self.o_norm(y.reshape(B * T * H, D_v)).reshape(B, T, H, D_v)
        y = (y * gate).reshape(B, T, H * D_v)
        out = self.o_proj(y)

        if cache is not None:
            cache.state = (S, new_conv_ctx)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Embedding  (attention layers only)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 500000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self._inv_freq = inv_freq

    def __call__(self, seq_len: int, offset: int = 0):
        t    = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freq = mx.outer(t, self._inv_freq)
        emb  = mx.concatenate([freq, freq], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    @staticmethod
    def apply(q, k, cos, sin):
        """q, k: (B, H, T, D);  cos/sin: (T, D)"""
        def _rot(x):
            h = x.shape[-1] // 2
            return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)
        cos = cos[None, None]
        sin = sin[None, None]
        return q * cos + _rot(q) * sin, k * cos + _rot(k) * sin


# ─────────────────────────────────────────────────────────────────────────────
# Full Attention  (full_attention layers)
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Causal MHA with RoPE and OLMo2-style QK-norm.

    Post-norm: norm is applied to the mixer output in HybridLayer, not inside
    here. QK-norm is applied on the full (H*D) projection before reshape.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        H   = args.num_attention_heads
        Hkv = args.num_key_value_heads
        D   = args.head_dim
        hid = args.hidden_size

        self.num_heads    = H
        self.num_kv_heads = Hkv
        self.head_dim     = D
        self.scale        = D ** -0.5

        self.q_proj = nn.Linear(hid, H * D,   bias=False)
        self.k_proj = nn.Linear(hid, Hkv * D, bias=False)
        self.v_proj = nn.Linear(hid, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, hid,   bias=False)

        # QK-norm on full projection (H*D, not per-head D)
        self.q_norm = RMSNorm(H * D,   eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(Hkv * D, eps=args.rms_norm_eps)

    def __call__(
        self,
        x:     mx.array,
        mask:  Optional[mx.array],
        cos:   mx.array,
        sin:   mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_norm(self.q_proj(x)).reshape(B, T, H,   D).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x)).reshape(B, T, Hkv, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x)             .reshape(B, T, Hkv, D).transpose(0, 2, 1, 3)

        q, k = RotaryEmbedding.apply(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if Hkv < H:
            k = mx.repeat(k, H // Hkv, axis=1)
            v = mx.repeat(v, H // Hkv, axis=1)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask

        out = mx.matmul(mx.softmax(scores, axis=-1), v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Decoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class HybridLayer(nn.Module):
    """
    One decoder layer. Norm structure differs by type:

    linear_attention (GDN) — pre-norm:
        x = x + GDN(input_layernorm(x))
        x = x + FFN(post_attention_layernorm(x))

    full_attention — post-norm (OLMo2):
        x = x + post_attention_layernorm(Attn(x))
        x = x + post_feedforward_layernorm(FFN(x))
    """

    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        assert layer_type in ("linear_attention", "full_attention")
        self.layer_type = layer_type

        if layer_type == "linear_attention":
            self.input_layernorm          = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mixer = GatedDeltaNet(args)
        else:
            self.post_attention_layernorm  = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mixer = Attention(args)

        self.mlp = SwiGLU(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x:     mx.array,
        mask:  Optional[mx.array],
        cos:   mx.array,
        sin:   mx.array,
        cache=None,  # GDNCache | KVCache | None
    ) -> mx.array:
        if self.layer_type == "linear_attention":
            # mx.checkpoint avoids storing GDN's intermediate tensors on the autograd tape.
            # The T-step recurrence accumulates O(T) intermediate S/outer tensors (~1.1MB each).
            # Without checkpoint, MLX keeps all of these on the tape until backward completes —
            # ~3.4GB at T=64, ~13.4GB at T=256 (24 layers × T steps × 2 tensors × 1.1MB).
            # mx.checkpoint only keeps input/output on the tape; intermediates are freed.
            # stop_gradient ensures the backward never triggers a checkpoint recompute,
            # so we get full memory savings with zero extra compute cost.
            normed = self.input_layernorm(x)
            def _gdn_fwd(n):
                return self.mixer(n, cache=cache)
            x = x + mx.stop_gradient(mx.checkpoint(_gdn_fwd)(normed))
            x = x + self.mlp(self.post_attention_layernorm(x))
        else:
            x = x + self.post_attention_layernorm(
                self.mixer(x, mask=mask, cos=cos, sin=sin, cache=cache)
            )
            x = x + self.post_feedforward_layernorm(self.mlp(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model  (mlx-lm entry point — class MUST be named "Model")
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    OLMo Hybrid 7B.

    mlx-lm interface:
        logits = model(input_ids, cache=cache)    # (B, T, vocab)
        cache  = model.make_cache()               # one entry per layer
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm         = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head      = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        self.layers = [
            HybridLayer(args, args.layer_types[i])
            for i in range(args.num_hidden_layers)
        ]

        self.rotary = RotaryEmbedding(
            head_dim=args.head_dim,
            base=args.rope_theta,
        )

    def __call__(
        self,
        inputs: mx.array,                  # (B, T) int32
        cache:  Optional[List] = None,     # list[GDNCache | KVCache | None]
    ) -> mx.array:                         # (B, T, vocab_size)
        B, T = inputs.shape
        x = self.embed_tokens(inputs)

        # RoPE offset = number of tokens already in the KV cache
        # (use the first attention-layer cache to read offset)
        attn_offset = 0
        if cache is not None:
            for c in cache:
                if isinstance(c, KVCache):
                    attn_offset = c.offset
                    break

        cos, sin = self.rotary(T, offset=attn_offset)

        # Causal mask (only needed during prefill; decode is T=1)
        mask = None
        if T > 1:
            total = T + attn_offset
            mask  = mx.triu(mx.full((T, total), float("-inf")), k=1 + attn_offset)
            mask  = mask[None, None].astype(x.dtype)

        if cache is None:
            cache = [None] * self.args.num_hidden_layers

        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask, cos=cos, sin=sin, cache=cache[i])

        x = self.norm(x)
        return self.lm_head(x)

    def make_cache(self) -> List:
        """
        Create per-layer cache objects. Called by mlx-lm's make_prompt_cache().
        Returns a mixed list: GDNCache for linear_attention, KVCache for full_attention.
        """
        caches = []
        for lt in self.args.layer_types:
            if lt == "linear_attention":
                caches.append(GDNCache())
            else:
                caches.append(KVCache(
                    head_dim=self.args.head_dim,
                    n_heads=self.args.num_key_value_heads,
                ))
        return caches

    # Properties expected by mlx-lm utilities
    @property
    def head_dim(self) -> int:
        return self.args.head_dim

    @property
    def n_kv_heads(self) -> int:
        return self.args.num_key_value_heads


# ─────────────────────────────────────────────────────────────────────────────
# Weight sanitization  (replaces convert.py in the mlx-lm load pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(weights: dict) -> dict:
    """
    Map HuggingFace safetensors weight names → MLX model attribute paths.
    Called by mlx-lm's load() after reading the checkpoint shards.

    Mapping rules
    ─────────────
    model.embed_tokens.weight          → embed_tokens.weight
    model.norm.weight                  → norm.weight
    lm_head.weight                     → lm_head.weight

    GDN (linear_attention) layers:
      model.layers.{i}.input_layernorm.*        → layers.{i}.input_layernorm.*
      model.layers.{i}.post_attention_layernorm.* → layers.{i}.post_attention_layernorm.*
      model.layers.{i}.linear_attn.*           → layers.{i}.mixer.*
      model.layers.{i}.mlp.*                   → layers.{i}.mlp.*

    Attention (full_attention) layers:
      model.layers.{i}.post_attention_layernorm.*  → layers.{i}.post_attention_layernorm.*
      model.layers.{i}.post_feedforward_layernorm.* → layers.{i}.post_feedforward_layernorm.*
      model.layers.{i}.self_attn.*                → layers.{i}.mixer.*
      model.layers.{i}.mlp.*                      → layers.{i}.mlp.*

    ShortConv weight reshape:
      HF stores (C, 1, K)  [PyTorch: (out_ch, in_ch/groups, kW)]
      MLX needs  (C, K, 1)  [MLX conv1d: (out_ch, kW, in_ch/groups)]
      Transform: w.transpose(0, 2, 1)
    """
    out = {}
    for hf_name, w in weights.items():
        mlx_name = _hf_to_mlx(hf_name)
        if mlx_name is None:
            continue

        # Reshape depthwise conv weights: (C, 1, K) → (C, K, 1)
        if "conv1d.weight" in mlx_name and hasattr(w, "ndim") and w.ndim == 3 and w.shape[1] == 1:
            w = w.transpose(0, 2, 1)

        out[mlx_name] = w
    return out


def _hf_to_mlx(hf_name: str) -> Optional[str]:
    """Return the MLX attribute path for an HF weight name, or None to skip."""
    name = hf_name.removeprefix("model.")

    # Top-level weights
    if name == "embed_tokens.weight": return "embed_tokens.weight"
    if name == "norm.weight":         return "norm.weight"
    if hf_name == "lm_head.weight":   return "lm_head.weight"

    # Per-layer weights
    m = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if not m:
        return None

    i    = m.group(1)
    rest = m.group(2)

    # Norms and FFN (shared across both layer types)
    if rest in (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ):
        return f"layers.{i}.{rest}"

    if rest.startswith("mlp."):
        return f"layers.{i}.{rest}"

    # GDN (linear_attention) layers — HF prefix: linear_attn.*
    if rest.startswith("linear_attn."):
        sub = rest[len("linear_attn."):]
        return f"layers.{i}.mixer.{sub}"

    # Attention (full_attention) layers — HF prefix: self_attn.*
    if rest.startswith("self_attn."):
        sub = rest[len("self_attn."):]
        return f"layers.{i}.mixer.{sub}"

    return None  # unknown — skip
