"""
generate.py  —  Inference script for OLMo Hybrid 7B (MLX)

Usage:
    python generate.py \
        --weights ./weights/weights.npz \
        --prompt "The capital of France is" \
        --max-tokens 200 \
        --temperature 0.8

    python generate.py \
        --hf-dir ~/.cache/huggingface/hub/models--allenai--OLMo-Hybrid-7B/snapshots/main \
        --prompt "Language modeling is"

    # With LoRA fine-tuned weights
    python generate.py \
        --weights ./weights/weights.npz \
        --lora ./lora-weights/final.npz \
        --prompt "Explain what a transformer is in simple terms."

NOTE: Base model only — use completion-style prompts, not questions.
      With LoRA weights loaded, question-style prompts work fine.
"""

import argparse
import time
import numpy as np
import mlx.core as mx

from model import ModelArgs, Model
from train import inject_lora, load_lora


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading
# ─────────────────────────────────────────────────────────────────────────────

def load_weights_npz(model: Model, path: str):
    """Load weights from the .npz produced by convert.py."""
    print(f"Loading weights from {path} …")
    data = np.load(path)
    weights = {}
    for k in data.files:
        w = mx.array(data[k])
        # Saved weights have ShortConv shape (C, K); model expects (C, K, 1)
        if "conv1d.weight" in k and w.ndim == 2:
            w = w[:, :, None]
        weights[k] = w
    print(f"  {len(weights)} tensors loaded")
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print("  Weights applied ✓")


def load_weights_hf(model: Model, hf_dir: str):
    """Load directly from a HuggingFace safetensors checkpoint."""
    import json
    from pathlib import Path
    from safetensors import safe_open

    hf_dir = Path(hf_dir)
    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]

    raw = {}
    for shard in shard_names:
        with safe_open(str(hf_dir / shard), framework="numpy") as f:
            for key in f.keys():
                raw[key] = mx.array(f.get_tensor(key).astype(np.float16))

    print(f"  {len(raw)} tensors loaded")
    model.load_weights(list(raw.items()))
    mx.eval(model.parameters())
    print("  Weights applied ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str = "allenai/OLMo-Hybrid-7B"):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"Tokenizer load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample(logits: mx.array, temperature: float, top_p: float) -> int:
    if temperature == 0:
        return int(mx.argmax(logits, axis=-1))

    logits_np = np.array(logits[0], dtype=np.float32) / temperature
    logits_np -= logits_np.max()
    probs = np.exp(logits_np)
    probs /= probs.sum()

    if top_p < 1.0:
        idx = np.argsort(-probs)
        srt = probs[idx]
        cum = np.cumsum(srt)
        cut = np.searchsorted(cum, top_p) + 1
        srt[cut:] = 0.0
        srt /= srt.sum()
        return int(np.random.choice(idx, p=srt))

    return int(np.random.choice(len(probs), p=probs))


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    model:          Model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_p:          float = 0.95,
):
    ids       = tokenizer.encode(prompt) if tokenizer else [ord(c) for c in prompt]
    input_ids = mx.array([ids], dtype=mx.int32)

    print(f"\nPrompt ({len(ids)} tokens): {repr(prompt)}\n")

    # Prefill — build cache from the full prompt
    t0    = time.time()
    cache = model.make_cache()
    logits = model(input_ids, cache=cache)   # (1, T, vocab)
    mx.eval(logits)
    prefill_tps = len(ids) / (time.time() - t0)

    # Decode
    generated = []
    next_id   = _sample(logits[:, -1, :], temperature, top_p)
    generated.append(next_id)

    print(prompt, end="", flush=True)
    if tokenizer:
        print(tokenizer.decode([next_id]), end="", flush=True)

    t1 = time.time()
    for _ in range(max_new_tokens - 1):
        x      = mx.array([[next_id]], dtype=mx.int32)
        logits = model(x, cache=cache)       # cache updated in-place each step
        mx.eval(logits)
        next_id = _sample(logits[:, -1, :], temperature, top_p)
        generated.append(next_id)

        if tokenizer:
            print(tokenizer.decode([next_id]), end="", flush=True)

        if tokenizer and next_id == tokenizer.eos_token_id:
            break

    decode_tps = len(generated) / max(time.time() - t1, 1e-6)

    print(f"\n\n── Stats ───────────────────────────────────")
    print(f"  Prefill : {len(ids)} tokens @ {prefill_tps:.1f} tok/s")
    print(f"  Decode  : {len(generated)} tokens @ {decode_tps:.1f} tok/s")

    return tokenizer.decode(generated) if tokenizer else generated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLMo Hybrid 7B — MLX inference")
    parser.add_argument("--weights",     help="Path to weights.npz (convert.py output)")
    parser.add_argument("--hf-dir",      help="Path to HF checkpoint dir (safetensors)")
    parser.add_argument("--prompt",      default="The capital of France is")
    parser.add_argument("--max-tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p",       type=float, default=0.95)
    parser.add_argument("--hf-model",    default="allenai/OLMo-Hybrid-7B",
                        help="HF model ID for tokenizer")
    parser.add_argument("--lora",        default=None,
                        help="Path to LoRA weights .npz (from train.py)")
    parser.add_argument("--lora-rank",   type=int, default=8)
    args = parser.parse_args()

    if not args.weights and not args.hf_dir:
        parser.error("Provide --weights (npz) or --hf-dir (safetensors)")

    mx.set_default_device(mx.gpu)

    cfg   = ModelArgs()
    model = Model(cfg)

    if args.weights:
        load_weights_npz(model, args.weights)
    else:
        load_weights_hf(model, args.hf_dir)

    if args.lora:
        inject_lora(model, rank=args.lora_rank)
        load_lora(model, args.lora)
        print(f"LoRA weights loaded from {args.lora}")

    tokenizer = load_tokenizer(args.hf_model)

    generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
