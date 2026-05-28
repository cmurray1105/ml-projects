"""
generate.py  —  Inference script for OLMo Hybrid 7B (MLX)

Usage:
    # Single prompt
    python generate.py \
        --weights ./weights/weights.npz \
        --prompt "The capital of France is"

    # Interactive chat with LoRA weights
    python generate.py \
        --weights ./weights/weights.npz \
        --lora ./lora-hermes-overnight/final.npz \
        --chat

    # Chat with custom system prompt
    python generate.py \
        --weights ./weights/weights.npz \
        --lora ./lora-hermes-overnight/final.npz \
        --chat \
        --system "You are a FINRA compliance expert at Red Oak. Answer questions about securities regulations clearly and concisely."

Commands inside chat:
    /reset   — clear conversation history (keep system prompt)
    /system  — change system prompt mid-session
    /quit    — exit
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

def load_tokenizer(model_id: str = "allenai/OLMo-Hybrid-7B",
                   local_tokenizer: str = "./tokenizer-chatml"):
    """
    Load tokenizer, preferring the local copy that has the ChatML template baked in.
    Falls back to HuggingFace if the local path doesn't exist.
    """
    from pathlib import Path
    try:
        from transformers import AutoTokenizer
        path = local_tokenizer if Path(local_tokenizer).exists() else model_id
        tok = AutoTokenizer.from_pretrained(path)
        if tok.chat_template:
            print(f"  Tokenizer loaded from {path} (chat_template ✓)")
        else:
            print(f"  Tokenizer loaded from {path} (no chat_template)")
        return tok
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
# Chat helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_chat(messages: list, system_prompt: str = None, tokenizer=None) -> str:
    """
    Build a ChatML prompt string from a message history.
    Uses tokenizer.apply_chat_template() if the tokenizer has a template;
    falls back to manual ChatML construction otherwise.
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Manual fallback
    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def chat(
    model:          Model,
    tokenizer,
    system_prompt:  str   = None,
    max_new_tokens: int   = 512,
    temperature:    float = 0.8,
    top_p:          float = 0.95,
):
    """Interactive chat loop with history, /reset, /system, /quit commands."""

    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>") if tokenizer else None
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") if tokenizer else None
    history   = []

    print("\n─────────────────────────────────────────────")
    print("  Rocco  —  OLMo Hybrid 7B")
    print("  /reset   clear history   /system  new system prompt   /quit  exit")
    if system_prompt:
        print(f"  System: {system_prompt[:80]}{'…' if len(system_prompt) > 80 else ''}")
    print("─────────────────────────────────────────────\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.split(None, 1)
            if cmd[0] == "/quit":
                print("Bye!")
                break
            elif cmd[0] == "/reset":
                history = []
                print("  [history cleared]\n")
                continue
            elif cmd[0] == "/system":
                if len(cmd) > 1:
                    system_prompt = cmd[1]
                    history = []          # reset history so new sys prompt takes effect
                    print(f"  [system prompt updated; history cleared]\n")
                else:
                    print("  Usage: /system <new system prompt>\n")
                continue
            else:
                print(f"  Unknown command: {cmd[0]}\n")
                continue

        # ── Build prompt ───────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        prompt = format_chat(history, system_prompt, tokenizer=tokenizer)

        ids       = tokenizer.encode(prompt) if tokenizer else [ord(c) for c in prompt]
        input_ids = mx.array([ids], dtype=mx.int32)

        # Prefill
        cache  = model.make_cache()
        logits = model(input_ids, cache=cache)
        mx.eval(logits)

        # Decode — stream tokens until EOS or <|im_end|>
        print("\nRocco: ", end="", flush=True)
        generated    = []
        next_id      = _sample(logits[:, -1, :], temperature, top_p)
        reply_tokens = []

        t0 = time.time()
        for _ in range(max_new_tokens):
            generated.append(next_id)

            # Stop conditions — EOS, <|im_end|>, or start of a new turn
            if tokenizer and (next_id == tokenizer.eos_token_id
                              or next_id == im_end_id
                              or next_id == im_start_id):
                break

            tok_str = tokenizer.decode([next_id]) if tokenizer else chr(next_id)
            print(tok_str, end="", flush=True)
            reply_tokens.append(next_id)

            x      = mx.array([[next_id]], dtype=mx.int32)
            logits = model(x, cache=cache)
            mx.eval(logits)
            next_id = _sample(logits[:, -1, :], temperature, top_p)

        elapsed = max(time.time() - t0, 1e-6)
        print(f"\n  [{len(generated)} tokens @ {len(generated)/elapsed:.1f} tok/s]\n")

        # Store assistant reply (decoded, strip trailing im_end if present)
        reply_text = tokenizer.decode(reply_tokens).rstrip("<|im_end|>").strip() if tokenizer else ""
        history.append({"role": "assistant", "content": reply_text})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLMo Hybrid 7B — MLX inference")
    parser.add_argument("--weights",     help="Path to weights.npz (convert.py output)")
    parser.add_argument("--hf-dir",      help="Path to HF checkpoint dir (safetensors)")
    parser.add_argument("--prompt",      default="The capital of France is")
    parser.add_argument("--max-tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p",       type=float, default=0.95)
    parser.add_argument("--hf-model",    default="allenai/OLMo-Hybrid-7B",
                        help="HF model ID for tokenizer")
    parser.add_argument("--lora",        default=None,
                        help="Path to LoRA weights .npz (from train.py)")
    parser.add_argument("--lora-rank",   type=int, default=8)
    parser.add_argument("--chat",        action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--system",      default=None,
                        help="System prompt for chat mode")
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

    if args.chat:
        chat(
            model,
            tokenizer,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
