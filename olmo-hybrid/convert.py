"""
convert.py  —  Convert allenai/Olmo-Hybrid-7B HuggingFace weights → MLX npz

Usage:
    # 1. Download (first time, ~14GB):
    python convert.py --download --hf-model allenai/Olmo-Hybrid-7B --out ./weights

    # 2. Convert only (if already downloaded):
    python convert.py --hf-dir ~/.cache/huggingface/hub/models--allenai--Olmo-Hybrid-7B/... \
                      --out ./weights

    # 3. Probe weight names only (no conversion):
    python convert.py --probe --hf-model allenai/Olmo-Hybrid-7B

The converter maps HuggingFace weight names to our MLX model's attribute paths.

HF naming (inferred from transformers source + NVLabs GDN structure):
  model.embed_tokens.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.post_attention_layernorm.weight
  model.layers.{i}.mlp.gate_proj.weight
  model.layers.{i}.mlp.up_proj.weight
  model.layers.{i}.mlp.down_proj.weight

  GDN layers:
  model.layers.{i}.gdn.q_proj.weight
  model.layers.{i}.gdn.k_proj.weight
  model.layers.{i}.gdn.v_proj.weight
  model.layers.{i}.gdn.b_proj.weight       ← beta
  model.layers.{i}.gdn.a_proj.weight       ← alpha
  model.layers.{i}.gdn.g_proj.weight       ← gate
  model.layers.{i}.gdn.o_proj.weight
  model.layers.{i}.gdn.q_conv1d.weight
  model.layers.{i}.gdn.k_conv1d.weight
  model.layers.{i}.gdn.v_conv1d.weight
  model.layers.{i}.gdn.o_norm.weight

  Attention layers:
  model.layers.{i}.self_attn.q_proj.weight
  model.layers.{i}.self_attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight
  model.layers.{i}.self_attn.o_proj.weight

  model.norm.weight
  lm_head.weight

MLX attribute paths (matching our model.py):
  embed_tokens.weight
  layers.{i}.input_layernorm.weight
  layers.{i}.post_attention_layernorm.weight
  layers.{i}.mlp.gate_proj.weight
  layers.{i}.mlp.up_proj.weight
  layers.{i}.mlp.down_proj.weight

  GDN:
  layers.{i}.mixer.q_proj.weight
  layers.{i}.mixer.k_proj.weight
  layers.{i}.mixer.v_proj.weight
  layers.{i}.mixer.b_proj.weight
  layers.{i}.mixer.a_proj.weight
  layers.{i}.mixer.g_proj.weight
  layers.{i}.mixer.o_proj.weight
  layers.{i}.mixer.q_conv1d.weight    ← depthwise, shape may need reshape
  layers.{i}.mixer.k_conv1d.weight
  layers.{i}.mixer.v_conv1d.weight
  layers.{i}.mixer.o_norm.weight

  Attention:
  layers.{i}.mixer.q_proj.weight
  layers.{i}.mixer.k_proj.weight
  layers.{i}.mixer.v_proj.weight
  layers.{i}.mixer.o_proj.weight

  norm.weight
  lm_head.weight
"""

import argparse
import json
import re
import os
import ml_dtypes  # registers bfloat16 as a numpy dtype — must import before safetensors
import numpy as np
from pathlib import Path

HYBRID_RATIO = 3   # 3 GDN layers per 1 Attention layer

def layer_type(layer_idx: int, hybrid_ratio: int = HYBRID_RATIO) -> str:
    if (layer_idx + 1) % (hybrid_ratio + 1) == 0:
        return "attn"
    return "gdn"


# ─────────────────────────────────────────────────────────────────────────────
# Weight name mapping
# ─────────────────────────────────────────────────────────────────────────────

def hf_to_mlx_name(hf_name: str, num_layers: int = 32) -> str | None:
    """
    Map a HuggingFace weight name → MLX model attribute path.
    Returns None if the weight should be skipped.

    Actual HF names confirmed from inspect_weights.py output:

    GDN (linear_attention) layers:
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.linear_attn.{q,k,v}_proj.weight
        model.layers.{i}.linear_attn.{q,k,v}_conv1d.weight
        model.layers.{i}.linear_attn.{a,b,g}_proj.weight
        model.layers.{i}.linear_attn.o_proj.weight
        model.layers.{i}.linear_attn.o_norm.weight
        model.layers.{i}.linear_attn.A_log          ← not a .weight suffix
        model.layers.{i}.linear_attn.dt_bias         ← not a .weight suffix
        model.layers.{i}.mlp.{gate,up,down}_proj.weight

    Attention (full_attention) layers:
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.post_feedforward_layernorm.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.self_attn.{q,k}_norm.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        (no input_layernorm)
    """

    # Strip 'model.' prefix
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]

    # Top-level
    if name == "embed_tokens.weight":
        return "embed_tokens.weight"
    if name == "norm.weight":
        return "norm.weight"
    if hf_name == "lm_head.weight":
        return "lm_head.weight"

    # Layer weights
    m = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if not m:
        return None

    i = int(m.group(1))
    rest = m.group(2)
    ltype = layer_type(i)

    # ── Shared (both layer types) ─────────────────────────────────────────
    if rest in ("input_layernorm.weight",
                "post_attention_layernorm.weight",
                "post_feedforward_layernorm.weight"):
        return f"layers.{i}.{rest}"
    if rest.startswith("mlp."):
        return f"layers.{i}.{rest}"

    # ── GDN (linear_attention) layers ─────────────────────────────────────
    # HF prefix: linear_attn.*
    if ltype == "gdn" and rest.startswith("linear_attn."):
        sub = rest[len("linear_attn."):]
        # A_log and dt_bias are stored without ".weight" suffix in HF
        # but we map them to the same attribute path in our model
        return f"layers.{i}.mixer.{sub}"

    # ── Attention (full_attention) layers ─────────────────────────────────
    # HF prefix: self_attn.*
    if ltype == "attn" and rest.startswith("self_attn."):
        sub = rest[len("self_attn."):]
        return f"layers.{i}.mixer.{sub}"

    return None   # unknown / skip


def reshape_conv_weight(w: np.ndarray) -> np.ndarray:
    """
    HF ShortConvolution weight is stored as (channels, 1, kernel_size)
    (standard depthwise conv format).  Our ShortConv uses (channels, kernel_size).
    """
    if w.ndim == 3 and w.shape[1] == 1:
        return w[:, 0, :]   # (C, 1, K) → (C, K)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Safetensors loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_hf_weights(model_dir: str):
    """List all tensors in a HF safetensors checkpoint directory."""
    model_dir = Path(model_dir)

    # Try index file first
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        return sorted(idx["weight_map"].keys()), idx["weight_map"]

    # Single shard
    shard_path = model_dir / "model.safetensors"
    if shard_path.exists():
        from safetensors import safe_open
        with safe_open(str(shard_path), framework="numpy") as f:
            names = sorted(f.keys())
        return names, {n: "model.safetensors" for n in names}

    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def load_tensor(model_dir: str, filename: str, key: str) -> np.ndarray:
    from safetensors import safe_open
    path = Path(model_dir) / filename
    with safe_open(str(path), framework="numpy") as f:
        return f.get_tensor(key)  # ml_dtypes makes bfloat16 a valid numpy dtype


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(hf_dir: str, out_dir: str, num_layers: int = 32):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Listing weights from {hf_dir}…")
    all_names, file_map = list_hf_weights(hf_dir)

    mlx_weights = {}
    skipped = []
    unknown = []

    for hf_name in all_names:
        mlx_name = hf_to_mlx_name(hf_name, num_layers=num_layers)
        if mlx_name is None:
            unknown.append(hf_name)
            continue

        filename = file_map[hf_name]
        w = load_tensor(hf_dir, filename, hf_name).astype(np.float16)

        # Reshape conv weights
        if "conv1d.weight" in mlx_name:
            w = reshape_conv_weight(w)

        # Linear weights in HF are (out, in); MLX nn.Linear expects (out, in) too.
        # No transpose needed.

        mlx_weights[mlx_name] = w
        print(f"  {hf_name}  →  {mlx_name}  {w.shape}")

    # tie_word_embeddings = False for OLMo Hybrid — lm_head is a separate tensor.
    # If it's still missing (unexpected), warn rather than silently alias.
    if "lm_head.weight" not in mlx_weights:
        print("  ⚠️  lm_head.weight not found in converted weights!")
        print("      Check hf_to_mlx_name() — the HF name for lm_head may differ.")

    # Save
    save_path = out_path / "weights.npz"
    np.savez(str(save_path), **mlx_weights)
    print(f"\nSaved {len(mlx_weights)} tensors to {save_path}")

    if unknown:
        print(f"\n⚠️  {len(unknown)} unrecognised HF weight names (saved to unknown_weights.txt):")
        for n in unknown[:20]:
            print(f"    {n}")
        with open(out_path / "unknown_weights.txt", "w") as f:
            f.write("\n".join(unknown))

    # Also save config snapshot
    config_path = Path(hf_dir) / "config.json"
    if config_path.exists():
        import shutil
        shutil.copy(config_path, out_path / "config.json")

    return save_path


def probe(hf_model: str, num_layers: int = 32):
    """Download config + weight list, print names + proposed mapping.
    Handles both multi-shard (index.json) and single-shard models.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # Try index file first (multi-shard)
    try:
        idx_path = hf_hub_download(hf_model, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        hf_names = sorted(idx["weight_map"].keys())
    except Exception:
        # Single-shard: download and peek at keys
        print("No index file found — downloading single shard to probe keys…")
        shard_path = hf_hub_download(hf_model, "model.safetensors")
        with safe_open(shard_path, framework="numpy") as f:
            hf_names = sorted(f.keys())

    print(f"\n{'HF weight name':<70} {'MLX name'}")
    print("-" * 120)
    for hf_name in hf_names:
        mlx_name = hf_to_mlx_name(hf_name, num_layers=num_layers)
        status = mlx_name or "⚠️  UNMATCHED"
        print(f"{hf_name:<70} {status}")


def download_model(hf_model: str):
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        hf_model,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "*.ot", "*.onnx"],
    )
    print(f"\nDownloaded to: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OLMo Hybrid HF weights → MLX")
    parser.add_argument("--hf-model", default="allenai/Olmo-Hybrid-7B", help="HuggingFace model id")
    parser.add_argument("--hf-dir", default=None, help="Local path to HF model dir (skip download)")
    parser.add_argument("--out", default="./weights", help="Output dir for MLX weights")
    parser.add_argument("--download", action="store_true", help="Download model from HF hub")
    parser.add_argument("--probe", action="store_true", help="Print weight name mapping without converting")
    parser.add_argument("--num-layers", type=int, default=32)
    args = parser.parse_args()

    if args.probe:
        probe(args.hf_model, num_layers=args.num_layers)
    else:
        hf_dir = args.hf_dir
        if hf_dir is None:
            if args.download:
                hf_dir = download_model(args.hf_model)
            else:
                print("Specify --hf-dir or pass --download to fetch from HuggingFace.")
                import sys; sys.exit(1)

        convert(hf_dir, args.out, num_layers=args.num_layers)
