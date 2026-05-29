"""
train.py  —  LoRA instruction fine-tuning for OLMo Hybrid 7B (MLX)

Strategy:
  - LoRA adapters on attention layers only (q_proj, v_proj, o_proj)
  - GDN layers contribute to the forward pass but gradients are stopped there
  - Keeps memory well within 36GB on M4 Pro
  - Trains on Alpaca-format OR Hermes/ShareGPT-format instruction data

Usage:
    # Alpaca (default)
    python train.py --weights ./weights/weights.npz --output ./lora-weights

    # Hermes/ShareGPT (teknium/OpenHermes-2.5)
    python train.py --weights ./weights/weights.npz \\
        --hf-dataset teknium/OpenHermes-2.5 \\
        --dataset-format hermes \\
        --max-samples 100 \\
        --output ./lora-hermes-test

    # Train on a local JSON file (auto-detects format)
    python train.py --weights ./weights/weights.npz --data ./my_data.json --output ./lora-weights

    # Resume from a checkpoint
    python train.py --weights ./weights/weights.npz --lora-weights ./lora-weights/step-500.npz

Alpaca JSON format:
    [{"instruction": "...", "input": "", "output": "..."}, ...]

Hermes / ShareGPT format:
    [{"conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "..."},
        {"from": "gpt",    "value": "..."}
    ]}, ...]

After training, generate with LoRA weights:
    python generate.py --weights ./weights/weights.npz --lora ./lora-weights/final.npz --prompt "..."
"""

import argparse
import json
import math
import time
import random
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten

from model import ModelArgs, Model


# ─────────────────────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter.

    Forward: output = stop_gradient(base(x)) + (x @ A @ B) * scale
      - Base weights are always frozen via stop_gradient — no gradients flow there
      - Only A and B accumulate gradients
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        in_dim  = base.weight.shape[1]
        out_dim = base.weight.shape[0]

        # A: Kaiming uniform init (standard LoRA)
        # B: zeros — so the adapter starts as an identity delta (no initial perturbation)
        limit = math.sqrt(1.0 / in_dim)
        self.lora_a = mx.random.uniform(-limit, limit, (in_dim, rank))
        self.lora_b = mx.zeros((rank, out_dim))

    def __call__(self, x):
        # Base forward — stop_gradient ensures no grads flow into frozen base weights
        base_out = mx.stop_gradient(self.base(x))
        # LoRA delta
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + lora_out * self.scale

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base linear for faster inference."""
        delta = (self.lora_a @ self.lora_b).T * self.scale  # (out, in)
        merged_weight = self.base.weight + delta
        merged = nn.Linear(self.base.weight.shape[1], self.base.weight.shape[0], bias=False)
        merged.weight = merged_weight
        return merged


def inject_lora(model: Model, rank: int = 8, alpha: float = 16.0,
                targets: tuple = ("q_proj", "v_proj", "o_proj")):
    """
    Replace target projections in full_attention layers with LoRALinear wrappers.
    GDN layers are left untouched.
    """
    injected = 0
    for i, layer in enumerate(model.layers):
        if layer.layer_type == "full_attention":
            for proj_name in targets:
                original = getattr(layer.mixer, proj_name, None)
                if isinstance(original, nn.Linear):
                    setattr(layer.mixer, proj_name, LoRALinear(original, rank, alpha))
                    injected += 1

    print(f"  Injected LoRA into {injected} projections across "
          f"{sum(1 for l in model.layers if l.layer_type == 'full_attention')} attention layers")
    return model


def get_lora_params(model: Model):
    """Return a flat list of (name, array) for only LoRA parameters."""
    params = []
    for i, layer in enumerate(model.layers):
        if layer.layer_type == "full_attention":
            for proj_name in ("q_proj", "v_proj", "o_proj"):
                proj = getattr(layer.mixer, proj_name, None)
                if isinstance(proj, LoRALinear):
                    params.append((f"layers.{i}.mixer.{proj_name}.lora_a", proj.lora_a))
                    params.append((f"layers.{i}.mixer.{proj_name}.lora_b", proj.lora_b))
    return params


def count_params(model: Model):
    """Print trainable vs total parameter counts."""
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    lora  = sum(p.size for _, p in get_lora_params(model))
    print(f"  Total params   : {total / 1e6:.1f}M")
    print(f"  LoRA params    : {lora / 1e3:.1f}K  ({100 * lora / total:.3f}% of total)")


def save_lora(model: Model, path: str):
    """Save only LoRA adapter weights."""
    params = {k: np.array(v) for k, v in get_lora_params(model)}
    np.savez(path, **params)
    print(f"  LoRA weights saved → {path}  ({len(params)} tensors)")


def load_lora(model: Model, path: str):
    """Load LoRA adapter weights into an already-injected model."""
    data = np.load(path)
    for i, layer in enumerate(model.layers):
        if layer.layer_type == "full_attention":
            for proj_name in ("q_proj", "v_proj", "o_proj"):
                proj = getattr(layer.mixer, proj_name, None)
                if isinstance(proj, LoRALinear):
                    a_key = f"layers.{i}.mixer.{proj_name}.lora_a"
                    b_key = f"layers.{i}.mixer.{proj_name}.lora_b"
                    if a_key in data:
                        proj.lora_a = mx.array(data[a_key])
                    if b_key in data:
                        proj.lora_b = mx.array(data[b_key])
    print(f"  LoRA weights loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}
### Response:
"""

PROMPT_TEMPLATE_NO_INPUT = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def format_prompt(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return PROMPT_TEMPLATE.format(instruction=instruction, input_text=input_text)
    return PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction)


def detect_format(data: list) -> str:
    """Infer dataset format from the first example."""
    if not data:
        return "alpaca"
    first = data[0]
    if "conversations" in first:
        return "hermes"
    if "instruction" in first:
        return "alpaca"
    return "alpaca"


def load_dataset(
    data_path: str | None,
    hf_dataset: str = "tatsu-lab/alpaca",
    max_samples: int = None,
    dataset_format: str = "auto",
):
    """
    Load instruction-tuning data from a local JSON file or HuggingFace.

    Supports two formats (auto-detected unless --dataset-format is set):
      alpaca  — [{"instruction": "...", "input": "...", "output": "..."}]
      hermes  — [{"conversations": [{"from": "system/human/gpt", "value": "..."}]}]
    """
    if data_path:
        with open(data_path) as f:
            data = json.load(f)
        fmt = dataset_format if dataset_format != "auto" else detect_format(data)
        print(f"  Loaded {len(data)} examples from {data_path}  (format: {fmt})")
    else:
        try:
            from datasets import load_dataset as hf_load
            print(f"  Downloading {hf_dataset} from HuggingFace …")
            ds = hf_load(hf_dataset, split="train")

            # Infer format from dataset name if auto
            if dataset_format == "auto":
                fmt = "hermes" if "hermes" in hf_dataset.lower() or "sharegpt" in hf_dataset.lower() else "alpaca"
            else:
                fmt = dataset_format

            if fmt == "hermes":
                data = [{"conversations": r["conversations"]} for r in ds]
            else:
                data = [{"instruction": r["instruction"], "input": r.get("input", ""),
                         "output": r["output"]} for r in ds]

            print(f"  Loaded {len(data)} examples  (format: {fmt})")
        except ImportError:
            raise RuntimeError(
                "Install the `datasets` library: pip install datasets --break-system-packages\n"
                "Or provide a local JSON file with --data"
            )

    if max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    return data, fmt


def tokenize_example(item: dict, tokenizer, max_length: int):
    """
    Tokenize one Alpaca example. Returns (input_ids, labels) where labels
    has -100 for prompt tokens (ignored in loss) and real token ids for response.
    """
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    output      = item.get("output", "")

    prompt   = format_prompt(instruction, input_text)
    full     = prompt + output + tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids   = tokenizer.encode(full,   add_special_tokens=False)

    # Truncate to max_length
    full_ids = full_ids[:max_length]

    # Labels: -100 on prompt tokens, real ids on response tokens
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels   = [-100] * n_prompt + full_ids[n_prompt:]

    return full_ids, labels


def tokenize_hermes_example(item: dict, tokenizer, max_length: int):
    """
    Tokenize one Hermes/ShareGPT example using ChatML format.

    Input:  {"conversations": [{"from": "system/human/gpt", "value": "..."}]}

    ChatML template per turn:
        <|im_start|>{role}\\n{content}<|im_end|>\\n

    Loss masking:
        system / human turns → all -100 (model doesn't generate these)
        assistant turns      → header (-100) + content + <|im_end|> supervised
                               (model learns: given user message, generate this)

    Returns (input_ids, labels) or (None, None) if unusable.
    """
    conversations = item.get("conversations", [])
    if not conversations:
        return None, None

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}

    full_ids: list[int] = []
    labels:   list[int] = []

    for turn in conversations:
        role = role_map.get(turn.get("from", ""), None)
        if role is None:
            continue
        value = turn.get("value", "")

        header_text  = f"<|im_start|>{role}\n"
        footer_text  = "<|im_end|>\n"

        header_ids  = tokenizer.encode(header_text,  add_special_tokens=False)
        content_ids = tokenizer.encode(value,         add_special_tokens=False)
        footer_ids  = tokenizer.encode(footer_text,   add_special_tokens=False)

        turn_ids = header_ids + content_ids + footer_ids

        if role == "assistant":
            # Mask the role header; supervise the content + closing token
            turn_labels = [-100] * len(header_ids) + content_ids + footer_ids
        else:
            # System and human turns — fully masked, model doesn't generate these
            turn_labels = [-100] * len(turn_ids)

        full_ids.extend(turn_ids)
        labels.extend(turn_labels)

    # Truncate
    full_ids = full_ids[:max_length]
    labels   = labels[:max_length]

    if not full_ids or all(l == -100 for l in labels):
        return None, None

    return full_ids, labels


def make_batches(
    data,
    tokenizer,
    max_length: int,
    batch_size: int = 1,
    shuffle: bool = True,
    dataset_format: str = "alpaca",
    pad_to_length: Optional[int] = None,
):
    """
    Yield (input_ids, labels) as mx.arrays of shape (B, T).
    Examples within a batch are right-padded with 0 (input) / -100 (labels).

    pad_to_length: if set, all batches are padded to exactly this length.
      Required for mx.compile() — fixed shapes prevent graph recompilation
      every step. Without it, variable T causes recompilation per batch.
    """
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)

    buf_ids: list   = []
    buf_labels: list = []

    def _emit(ids_list, labels_list):
        T = pad_to_length if pad_to_length else max(len(x) for x in ids_list)
        padded_ids    = [x + [0]    * (T - len(x)) for x in ids_list]
        padded_labels = [x + [-100] * (T - len(x)) for x in labels_list]
        return (mx.array(padded_ids,    dtype=mx.int32),
                mx.array(padded_labels, dtype=mx.int32))

    _tokenize = (tokenize_hermes_example if dataset_format == "hermes"
                 else tokenize_example)

    for idx in indices:
        ids, labels = _tokenize(data[idx], tokenizer, max_length)
        if ids is None or all(l == -100 for l in labels):
            continue
        buf_ids.append(ids)
        buf_labels.append(labels)
        if len(buf_ids) == batch_size:
            yield _emit(buf_ids, buf_labels)
            buf_ids, buf_labels = [], []

    # Yield any leftover partial batch
    if buf_ids:
        yield _emit(buf_ids, buf_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def loss_fn(model: Model, input_ids: mx.array, labels: mx.array):
    """
    Next-token prediction loss, masked to response tokens only.

    input_ids: (1, T)
    labels:    (1, T) — -100 on prompt positions, real token ids on response

    Memory strategy:
      nn.checkpoint on the full forward pass discards ALL intermediate layer
      activations (residuals, MLP gate/up outputs, QKV tensors, etc.) from
      the forward pass.  Only the final logits tensor is kept.  During the
      backward pass, MLX re-runs the full forward to recompute whatever it
      needs — one layer at a time, not all simultaneously.

      Without this, all 32 layers' activations sit in Metal memory together
      during backward, adding ~4-8 GB on top of the 14.86 GB weights.
      With it, peak activation memory is O(1 layer) ≈ 10-20 MB.

      Cost: ≈2× forward compute per step.

      IMPORTANT: mx.checkpoint only differentiates w.r.t. the *explicit
      arguments* of the wrapped function. If the model is captured by closure,
      its parameters are invisible to checkpoint's VJP and receive ZERO
      gradient. We must thread model.trainable_parameters() through as an
      explicit input — this is exactly what nn.checkpoint does internally
      (unavailable in older MLX), so gradients flow to lora_a AND lora_b.
    """
    def _forward(params, ids):
        model.update(params)
        return model(ids, cache=None)

    logits = mx.checkpoint(_forward)(
        model.trainable_parameters(), input_ids
    )  # (B, T, vocab)

    # Shift: logits[t] predicts labels[t+1]
    shift_logits = logits[:, :-1, :]       # (B, T-1, vocab)
    shift_labels = labels[:, 1:]            # (B, T-1)

    # MLX doesn't support boolean indexing, so:
    # 1. Build a float mask (1.0 for response tokens, 0.0 for prompt/-100 / padding)
    # 2. Clamp -100 labels to 0 so cross_entropy doesn't get an invalid index
    # 3. Compute loss on all positions, zero out masked positions, average over valid count
    mask = (shift_labels != -100).astype(mx.float32)   # (B, T-1)

    n_valid = mask.sum()
    if n_valid == 0:
        return mx.array(0.0)

    safe_labels = mx.where(shift_labels == -100, mx.zeros_like(shift_labels), shift_labels)

    B, T_1, V = shift_logits.shape
    # cross_entropy expects (N, C) logits and (N,) labels
    loss_all = nn.losses.cross_entropy(
        shift_logits.reshape(B * T_1, V),
        safe_labels.reshape(B * T_1),
        reduction="none"
    ).reshape(B, T_1)                      # (B, T-1)
    loss = (loss_all * mask).sum() / n_valid
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:            Model,
    tokenizer,
    data:             list,
    output_dir:       str,
    num_epochs:       int   = 1,
    learning_rate:    float = 1e-4,
    max_length:       int   = 512,
    batch_size:       int   = 1,
    grad_accum_steps: int   = 4,
    save_every:       int   = 100,
    log_every:        int   = 1,
    dataset_format:   str   = "alpaca",
    use_compile:      bool  = False,
    max_steps:        int   = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    # value_and_grad differentiates w.r.t. model.trainable_parameters()
    # stop_gradient inside LoRALinear ensures only lora_a and lora_b accumulate grads
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # NOTE: mx.compile() is incompatible with this model — the GDN recurrence
    # calls mx.eval(S) inside the loop to manage memory, which is illegal inside
    # a compiled function. The --compile flag is accepted but has no effect.
    if use_compile:
        print("  ⚠  mx.compile() incompatible with GDN recurrence (mx.eval inside loop)")
        print("     Running without compile.")

    def loss_and_grad(input_ids, labels):
        return loss_value_and_grad(model, input_ids, labels)

    eff_batch = batch_size * grad_accum_steps
    print(f"\nTraining config:")
    print(f"  Examples     : {len(data)}")
    print(f"  Format       : {dataset_format}")
    print(f"  Epochs       : {num_epochs}")
    print(f"  Max seq len  : {max_length}")
    print(f"  Batch size   : {batch_size}  (micro-batch per forward pass)")
    print(f"  Grad accum   : {grad_accum_steps} steps  (effective batch size = {eff_batch})")
    print(f"  LR           : {learning_rate}")
    print(f"  Save every   : {save_every} optimizer steps")
    print(f"  Output dir   : {output_dir}\n")

    global_step  = 0   # optimizer update steps
    micro_step   = 0   # individual forward passes
    accum_grads  = None
    accum_loss   = 0.0
    t_start      = time.time()

    # ── Memory diagnostics ───────────────────────────────────────────────────
    print("\n── Memory Diagnostics ──────────────────────────────────────")
    try:
        def _mem():
            a = mx.get_active_memory() / 1e9
            c = mx.get_cache_memory()  / 1e9
            return f"active={a:.2f}GB  cache={c:.2f}GB"

        print(f"  Weights loaded:      {_mem()}")

        # Forward-only probe (no backward) to see baseline before training
        _probe_ids = mx.zeros((1, 64), dtype=mx.int32)
        _probe_logits = model(_probe_ids, cache=None)
        mx.eval(_probe_logits)
        print(f"  After forward T=64:  {_mem()}")
        del _probe_logits, _probe_ids
        mx.clear_cache()
        print(f"  After clear_cache:   {_mem()}")

    except Exception as e:
        _mem = None
        print(f"  (memory stats unavailable: {e})")

    for epoch in range(num_epochs):
        print(f"── Epoch {epoch + 1}/{num_epochs} ──────────────────────────")

        for input_ids, labels in make_batches(data, tokenizer, max_length,
                                               batch_size=batch_size,
                                               dataset_format=dataset_format,
                                               pad_to_length=max_length if use_compile else None):
            loss, grads = loss_and_grad(input_ids, labels)
            mx.eval(loss, grads)   # force evaluation now — prevents graph accumulation across micro-steps

            # Free MLX's metal buffer cache between micro-steps.
            # MLX keeps freed tensors in a cache pool to amortise malloc overhead.
            # During training this cache can grow to several GB without clearing.
            mx.clear_cache()

            loss_val = loss.item()   # pull scalar out of graph now (cheap)

            # First backward: print memory so we can see if QLoRA is working
            if micro_step == 0:
                mem_str = f"  [{_mem()}]" if _mem else ""
                print(f"  micro-step 0 backward OK — loss={loss_val:.4f}{mem_str}")

            # Skip NaN/Inf steps — check loss first, then grads
            if not math.isfinite(loss_val):
                micro_step += 1
                continue
            grad_leaves = [v for _, v in tree_flatten(grads)]
            if any(not mx.isfinite(g).all().item() for g in grad_leaves if isinstance(g, mx.array)):
                micro_step += 1
                continue

            # Accumulate gradients
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

            accum_loss += loss_val
            micro_step += 1

            # Optimizer step after accumulating enough gradients
            if micro_step % grad_accum_steps == 0:
                # Scale gradients by accumulation steps
                scale = 1.0 / grad_accum_steps
                accum_grads = tree_map(lambda g: g * scale, accum_grads)

                # Clip gradient norm (prevents explosion from fp16 scan matmuls)
                arrays = [v for _, v in tree_flatten(accum_grads) if isinstance(v, mx.array)]
                global_norm = mx.sqrt(sum(mx.sum(g * g) for g in arrays))
                mx.eval(global_norm)
                max_norm = 1.0
                clip_coef = float(min(1.0, max_norm / (global_norm.item() + 1e-6)))
                if clip_coef < 1.0:
                    accum_grads = tree_map(
                        lambda g: g * clip_coef if isinstance(g, mx.array) else g,
                        accum_grads
                    )

                optimizer.update(model, accum_grads)
                # Only eval trainable (LoRA) params — no need to re-eval frozen weights
                mx.eval(model.trainable_parameters(), optimizer.state)
                mx.clear_cache()

                global_step += 1
                avg_loss = accum_loss / grad_accum_steps

                # Reset accumulation
                accum_grads = None
                accum_loss  = 0.0

                if global_step % log_every == 0:
                    elapsed = time.time() - t_start
                    tok_s = (micro_step * batch_size * max_length) / max(elapsed, 0.001)
                    mem_str = f" | {_mem()}" if _mem else ""
                    print(f"  step {global_step:>5} | loss {avg_loss:.4f} | "
                          f"{tok_s:.0f} tok/s | {elapsed:.0f}s{mem_str}")

                if global_step % save_every == 0:
                    ckpt = output_dir / f"step-{global_step:06d}.npz"
                    save_lora(model, str(ckpt))

                if max_steps is not None and global_step >= max_steps:
                    print(f"\n  Reached --steps {max_steps}, stopping early.")
                    # Final save and exit
                    final_path = output_dir / "final.npz"
                    save_lora(model, str(final_path))
                    print(f"Training complete. Final weights → {final_path}")
                    return

    # Final save
    final_path = output_dir / "final.npz"
    save_lora(model, str(final_path))
    print(f"\nTraining complete. Final weights → {final_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading (reused from generate.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_base_weights(model: Model, path: str):
    print(f"Loading base weights from {path} …")
    data = np.load(path)
    weights = {}
    float32_count = 0
    for k in data.files:
        w = mx.array(data[k])
        # Saved weights have ShortConv shape (C, K); model expects (C, K, 1)
        if "conv1d.weight" in k and w.ndim == 2:
            w = w[:, :, None]
        # Cast float32 → float16.
        # HF bfloat16 weights are stored as float32 in .npz (numpy has no bfloat16).
        # Keeping them float32 doubles VRAM: 7.4B × 4B = 29.6GB vs 14.8GB for float16.
        if w.dtype == mx.float32:
            w = w.astype(mx.float16)
            float32_count += 1
        weights[k] = w
    print(f"  {len(weights)} tensors loaded", end="")
    if float32_count:
        print(f"  (cast {float32_count} float32 → float16 to halve VRAM)", end="")
    print()
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    # Report actual GPU usage after weights are loaded
    try:
        active_gb = mx.get_active_memory() / 1e9
        cache_gb  = mx.get_cache_memory()  / 1e9
        print(f"  GPU after load: active={active_gb:.2f}GB  cache={cache_gb:.2f}GB")
    except Exception:
        pass
    print("  Weights applied ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for OLMo Hybrid 7B")
    parser.add_argument("--weights",        required=True, help="Path to base weights.npz")
    parser.add_argument("--data",           default=None,  help="Local JSON file (Alpaca or Hermes format; omit to download from HF)")
    parser.add_argument("--hf-dataset",     default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-format", default="auto", choices=["auto", "alpaca", "hermes"],
                        help="Dataset format. 'auto' detects from content/dataset name (default)")
    parser.add_argument("--lora-weights",   default=None,  help="Resume from a LoRA checkpoint .npz")
    parser.add_argument("--output",       default="./lora-weights", help="Output directory for checkpoints")
    parser.add_argument("--hf-model",     default="allenai/OLMo-Hybrid-7B", help="HF model ID for tokenizer")
    parser.add_argument("--lora-rank",    type=int,   default=8)
    parser.add_argument("--lora-alpha",   type=float, default=16.0)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--max-length",   type=int,   default=256)
    parser.add_argument("--grad-accum",   type=int,   default=4)
    parser.add_argument("--save-every",   type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=1,    help="Examples per forward pass (pad-collated). Start at 2-4.")
    parser.add_argument("--log-every",    type=int,   default=1,    help="Print loss every N optimizer steps (default 1 = every step)")
    parser.add_argument("--max-samples",  type=int,   default=None, help="Cap dataset size (useful for quick tests)")
    parser.add_argument("--compile",      action="store_true",      help="Use mx.compile() for faster training (requires fixed --max-length padding)")
    parser.add_argument("--steps",        type=int,   default=None, help="Stop after this many optimizer steps (overrides --epochs)")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    # Build model and load base weights
    print("\n── Model ───────────────────────────────────────────────────")
    cfg   = ModelArgs()
    model = Model(cfg)
    load_base_weights(model, args.weights)

    # Freeze ALL base weights before injecting LoRA.
    # Without this, MLX treats all 7.4B params as trainable and AdamW tries to
    # allocate moment tensors for all of them (~59GB). After freeze(), only the
    # newly-added lora_a / lora_b arrays (which are created AFTER freeze) are
    # trainable. Optimizer state drops from ~59GB to ~12MB.
    model.freeze()
    print("  Base model frozen ✓")

    # Inject LoRA
    print("\n── LoRA ────────────────────────────────────────────────────")
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # Explicitly unfreeze LoRA params and re-freeze base weights.
    # model.freeze() marks params frozen at that point in time. But inject_lora()
    # replaces nn.Linear modules with LoRALinear objects *after* freeze(), so the
    # parent-level frozen state may not propagate correctly to the new raw mx.array
    # lora_a/lora_b attributes. We manually unfreeze each LoRALinear and re-freeze
    # its base Linear to guarantee only lora_a/lora_b are in model.trainable_parameters().
    lora_count = 0
    for layer in model.layers:
        if layer.layer_type == "full_attention":
            for proj_name in ("q_proj", "v_proj", "o_proj"):
                proj = getattr(layer.mixer, proj_name, None)
                if proj is not None and isinstance(proj, LoRALinear):
                    proj.unfreeze()      # unfreeze LoRALinear + its children
                    proj.base.freeze()   # re-freeze base Linear weights
                    lora_count += 1
    print(f"  Unfroze {lora_count} LoRA adapters, re-froze base weights ✓")

    # Verify: should be ~1474.6K, NOT 7432.3M
    from mlx.utils import tree_flatten as _tf
    n_trainable = sum(p.size for _, p in _tf(model.trainable_parameters()))
    print(f"  Trainable params: {n_trainable / 1e3:.1f}K  (target: ~1,474.6K)")
    if n_trainable > 5_000_000:
        print("  ⚠  WARNING: Too many trainable params — freeze not working! Will OOM.")
        raise RuntimeError(
            f"Expected ~1.5M trainable params, got {n_trainable/1e6:.1f}M. "
            "Check LoRALinear freeze logic."
        )
    else:
        print("  Trainable param count looks correct ✓")

    count_params(model)

    # ── QLoRA: 4-bit quantize all frozen layers ──────────────────────────────
    # Problem: backward pass for a float16 7B model requires ~15GB on top of
    # the 14.86GB weights — total exceeds 36GB → OOM.
    #
    # Fix: quantize the frozen base model to 4-bit (NF4-style via MLX).
    # 7.4B params × 0.5 bytes ≈ 3.7GB instead of 14.86GB. Saves ~11GB.
    # LoRA adapters (lora_a, lora_b) stay in float16 — gradients still exact.
    # base(x) dequantizes on-the-fly during the forward pass, adds the LoRA
    # delta in float16. This is the standard QLoRA training setup.
    #
    # Note: LoRALinear is nn.Module not nn.Linear, so quantize skips the
    # wrapper itself. It DOES quantize LoRALinear.base (nn.Linear), which is
    # correct — quantizing the base attention weights is the QLoRA objective.
    print("\n── QLoRA (4-bit) ───────────────────────────────────────────")
    print("  Quantizing frozen layers → 4-bit  (14.86GB → ~3.7GB expected) …")

    nn.quantize(model, bits=4, group_size=64)
    mx.eval(model.parameters())

    # quantize() replaces nn.Linear with nn.QuantizedLinear (new param tensors:
    # weight/scales/biases). Re-freeze everything, then re-unfreeze LoRA only.
    model.freeze()

    lora_count_q = 0
    for layer in model.layers:
        if layer.layer_type == "full_attention":
            for proj_name in ("q_proj", "v_proj", "o_proj"):
                proj = getattr(layer.mixer, proj_name, None)
                if proj is not None and isinstance(proj, LoRALinear):
                    proj.unfreeze()      # lora_a + lora_b trainable
                    proj.base.freeze()   # QuantizedLinear frozen
                    lora_count_q += 1
    print(f"  Re-unfroze {lora_count_q} LoRA adapters ✓")

    n_trainable_q = sum(p.size for _, p in _tf(model.trainable_parameters()))
    print(f"  Trainable params after quantize: {n_trainable_q / 1e3:.1f}K  (target: ~1,474.6K)")
    if n_trainable_q > 5_000_000:
        raise RuntimeError(
            f"QLoRA freeze failed: {n_trainable_q / 1e6:.1f}M trainable — check LoRA unfreeze loop"
        )

    try:
        active_gb = mx.get_active_memory() / 1e9
        saving_gb = 14.86 - active_gb
        print(f"  GPU after 4-bit:  {active_gb:.2f}GB  (saved ~{saving_gb:.1f}GB vs float16)")
    except Exception:
        pass
    print("  QLoRA ready ✓")

    # Optionally resume from checkpoint
    if args.lora_weights:
        print(f"\n── Resuming from {args.lora_weights}")
        load_lora(model, args.lora_weights)

    # Tokenizer
    print("\n── Tokenizer ───────────────────────────────────────────────")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Dataset
    print("\n── Dataset ─────────────────────────────────────────────────")
    data, fmt = load_dataset(
        args.data,
        args.hf_dataset,
        max_samples=args.max_samples,
        dataset_format=args.dataset_format,
    )
    print(f"  Format: {fmt}")

    # Train
    print("\n── Training ────────────────────────────────────────────────")
    train(
        model            = model,
        tokenizer        = tokenizer,
        data             = data,
        output_dir       = args.output,
        num_epochs       = args.epochs,
        learning_rate    = args.lr,
        max_length       = args.max_length,
        batch_size       = args.batch_size,
        grad_accum_steps = args.grad_accum,
        save_every       = args.save_every,
        log_every        = args.log_every,
        dataset_format   = fmt,
        use_compile      = args.compile,
        max_steps        = args.steps,
    )
