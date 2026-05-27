# OLMo Hybrid 7B — MLX Port + LoRA Training

Unofficial MLX port of [allenai/OLMo-Hybrid-7B](https://huggingface.co/allenai/OLMo-Hybrid-7B) for Apple Silicon, with QLoRA fine-tuning support.

**Status:** Working inference + training. mlx-lm upstream PR open: [cmurray1105/mlx-lm#add-olmo-hybrid](https://github.com/cmurray1105/mlx-lm/tree/add-olmo-hybrid)

---

## What this is

OLMo Hybrid 7B uses a **3:1 GatedDeltaNet / Attention hybrid** architecture — 24 GDN layers and 8 standard attention layers, alternating in a fixed pattern. GDN layers replace the KV cache (which grows linearly with context length) with a fixed-size recurrent state matrix. Memory stays constant regardless of context.

This makes it particularly interesting on Apple Silicon, where unified memory is the constraint.

When I went to run it in May 2025, there was no MLX port. So I built one.

---

## Architecture at a glance

```
hidden_size:        3840
num_layers:         32  (24 GDN + 8 Attention)
layer pattern:      GDN, GDN, GDN, Attn  ×8
GDN heads:          30  (key_dim=96, val_dim=192)
Attention heads:    30  (head_dim=128, no GQA)
FFN:                SwiGLU, intermediate_size=11008
RoPE theta:         500000 (long-context)
Context length:     65536
```

GDN recurrence per token per head:
```
alpha  = exp(softplus(a_proj(x) + dt_bias) * -exp(A_log))   # forget gate ∈ (0,1]
beta   = sigmoid(b_proj(x))                                   # write scale
k, q   = l2_normalize(k_proj(x)), l2_normalize(q_proj(x))
v      = v_proj(x)

Sk     = S @ k          # read
delta  = v - Sk         # error
S      = alpha*S + beta*outer(delta, k)    # delta-rule update
y      = S @ q          # output
out    = o_proj(o_norm(y) * silu(g_proj(x)))
```

The state `S` is shape `(B, H, val_dim, key_dim)`. It never grows.

---

## What was nontrivial to port

Standard MLX ports (LLaMA, Mistral) are mostly weight remapping — `mlx-lm` handles the boilerplate. OLMo Hybrid required implementing from scratch:

**GatedDeltaNet** — a linear attention variant using the delta rule. The PyTorch version uses custom CUDA kernels. MLX needs a Python-level `for` loop over token positions during prefill.

**ShortConv** — causal depthwise conv1d (`kernel=4`) applied to q, k, v before the recurrence. MLX doesn't expose grouped convolutions, so implemented as a manual sliding-window multiply.

**Hybrid norm structure** — GDN layers use pre-norm (LLaMA style); attention layers use post-norm (OLMo 2 style). Both in the same model, alternating.

**Heterogeneous cache** — `GDNCache` carries `(S_matrix, conv_context)` per GDN layer; `KVCache` is a standard growing KV buffer for attention layers. Both thread through the generation loop simultaneously.

---

## Bugs found during porting

**QK norm shape** — Attention layers use per-head QK normalization. The norm weight is `(head_dim,)` = `(128,)`, applied *after* reshaping to `(B, T, H, head_dim)`. An early implementation applied it to the full flattened projection `(H * head_dim,)` = `(3840,)` *before* reshape. Loads without error. Produces wrong outputs silently.

**RoPE theta** — OLMo Hybrid uses `rope_theta = 500000` for long-context position encoding. GPT-2 era default is `10000`. Position encodings with the wrong theta look correct for short sequences and collapse at longer ones.

---

## mlx-lm contribution

After getting standalone inference working, I ported the implementation to [mlx-lm](https://github.com/ml-explore/mlx-lm) conventions so it can be used with standard `mlx_lm.generate` / `mlx_lm.convert`.

Key differences from the standalone `model.py`:

- Single file at `mlx_lm/models/olmo_hybrid.py`, auto-loaded by `model_type: "olmo_hybrid"` in config.json
- `ModelArgs` extends `BaseModelArgs`; all config fields with defaults so `from_dict()` works
- `sanitize()` as an instance method on `Model`, handling three weight formats: HF raw safetensors, pre-converted MLX safetensors, and old MLX .npz (no `model.` prefix)
- `KVCache.state` property for compatibility with `mlx_lm`'s generation loop
- Uses `initialize_rope()` utility instead of a hand-rolled `RotaryEmbedding`
- `ShortConv` uses `mx.conv1d` with `groups=C` instead of a manual loop

The PR is at `cmurray1105/mlx-lm`, branch `add-olmo-hybrid`.

```bash
# With the PR merged, convert and generate become standard:
mlx_lm.convert --hf-path allenai/OLMo-Hybrid-7B --mlx-path ./olmo-hybrid-mlx
mlx_lm.generate --model ./olmo-hybrid-mlx --prompt "Language modeling is"
```

---

## QLoRA fine-tuning

`train.py` implements LoRA instruction fine-tuning that fits in 36GB:

**What's trainable:** LoRA adapters on attention `q_proj`, `v_proj`, `o_proj` only. 1.47M parameters out of 7.4B.

**Why not GDN layers:** GDN forward pass runs a sequential Python loop over token positions. Backpropagating through it would require storing all intermediate states in memory — O(T) GDN states per GDN layer. At T=512 that's enormous. Instead, `stop_gradient` cuts backprop at the GDN boundary. GDN layers participate in forward but not backward.

**4-bit quantization (QLoRA):** Base model weights are quantized to 4-bit via `nn.quantize(model, bits=4, group_size=64)`, compressing 14.86GB → ~4.19GB. LoRA adapters stay in float16. The base linear dequantizes on-the-fly during forward.

```
Base model (frozen, 4-bit):  ~4.19 GB
LoRA adapters (float16):     ~12 MB
Optimizer state (AdamW):     ~24 MB
Peak training memory:        ~8–10 GB total
```

**Gradient checkpointing:** The full forward pass is wrapped in `mx.checkpoint()`, which discards all intermediate activations and recomputes them on the backward pass. Peak activation memory is O(1 layer) instead of O(32 layers).

**Supported datasets:**
- Alpaca format: `[{"instruction": "...", "input": "...", "output": "..."}]`
- Hermes / ShareGPT format: `[{"conversations": [{"from": "system/human/gpt", "value": "..."}]}]`

---

## Performance (M4 Pro, 36GB)

| Phase   | Tokens | Speed      |
|---------|--------|------------|
| Prefill | 9 tok  | ~16 tok/s  |
| Decode  | 28 tok | ~4.9 tok/s |

Decode speed reflects the fundamental nature of GDN: each token requires a sequential state update. A chunked parallel scan (similar to Mamba's `pscan`) would improve prefill significantly. Main known TODO.

---

## Setup

```bash
git clone https://github.com/cmurray1105/olmo-hybrid-mlx
cd olmo-hybrid-mlx
pip install -r requirements.txt
```

**Step 1 — Inspect weight names (optional)**
```bash
python inspect_weights.py
```

**Step 2 — Download and convert weights**
```bash
python convert.py --download --out ./weights
```

**Step 3 — Generate**
```bash
python generate.py \
  --weights ./weights/weights.npz \
  --prompt "Language modeling is" \
  --max-tokens 200 \
  --temperature 0.8
```

Prompting note: this is a base model. Use completion-style prompts.

---

## Fine-tuning

**Alpaca (default)**
```bash
python train.py \
  --weights ./weights/weights.npz \
  --output ./lora-weights
```

**Hermes / ShareGPT (recommended for instruction following)**
```bash
python train.py \
  --weights ./weights/weights.npz \
  --hf-dataset teknium/OpenHermes-2.5 \
  --dataset-format hermes \
  --max-samples 1000 \
  --max-length 512 \
  --output ./lora-hermes
```

**Resume from checkpoint**
```bash
python train.py \
  --weights ./weights/weights.npz \
  --lora-weights ./lora-hermes/step-000500.npz \
  --output ./lora-hermes-continued
```

---

## Files

| File | What it does |
|------|-------------|
| `model.py` | Standalone MLX model — GDN, Attention, ShortConv, hybrid cache |
| `train.py` | QLoRA fine-tuning: 4-bit base + LoRA adapters on attention layers |
| `generate.py` | Inference with optional LoRA weights |
| `convert.py` | HF safetensors → MLX .npz conversion |
| `test_mlxlm_port.py` | 7-check smoke test for the mlx-lm version |
| `inspect_weights.py` | Print HF weight names without downloading full model |
| `debug_scale.py` | Layer-by-layer activation scale debugging |

---

## Known limitations / TODO

- [ ] Sequential recurrence only — no chunked scan for fast prefill
- [ ] ShortConv `allow_neg_eigval` beta-scaling not validated against HF reference
- [ ] No evaluation harness integration
- [ ] Generated LoRA weights not yet benchmarked post-training

---

## Background reading

- [OLMo Hybrid 7B on HuggingFace](https://huggingface.co/allenai/OLMo-Hybrid-7B)
- [GatedDeltaNet paper (arXiv 2412.06464)](https://arxiv.org/abs/2412.06464)
- [NVLabs GatedDeltaNet reference implementation](https://github.com/NVlabs/GatedDeltaNet)
- [OLMo 2 technical report (arXiv 2501.00656)](https://arxiv.org/abs/2501.00656)
- [MLX documentation](https://ml-explore.github.io/mlx/)

---

*Apache 2.0. Base model weights from AllenAI under Apache 2.0.*
