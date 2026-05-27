# I LoRA Fine-Tuned a Hybrid Transformer/RNN Nobody Has Fine-Tuned Before. Here's What Broke.

*OLMo Hybrid 7B is a brand-new architecture combining two fundamentally different memory mechanisms in one model. There was no MLX port. No LoRA fine-tuning recipe. No QLoRA reference implementation. Just 36GB of unified memory, a lot of OOM errors, and a debugging journey that turned out to be surprisingly interesting.*

---

## The Setup

In early 2025, AllenAI released [OLMo Hybrid 7B](https://huggingface.co/allenai/OLMo-Hybrid-7B) — a 7 billion parameter language model with an unusual twist. It doesn't use pure transformer attention. Instead, 75% of its layers are something called **GatedDeltaNet**, and only 25% are standard attention. The architecture alternates in a 3:1 pattern: GDN, GDN, GDN, Attention — repeating 8 times for 32 layers total.

The reason this exists comes down to one word: **memory**. Standard transformers store a KV cache (a record of every token they've seen) that grows linearly with context length. At 65,536 tokens — OLMo Hybrid's target context window — that cache can consume more memory than the model weights themselves. GatedDeltaNet was designed to solve this.

I wanted to run this model on Apple Silicon, where I have 36GB of fast unified memory and no GPU rental bill. The problem: there was no MLX port. So I built one.

---

## GatedDeltaNet: The Idea

The core concept behind GDN is ancient — it's the **delta rule**, a learning algorithm published by Widrow and Hoff in 1960. The idea is simple: when you make a prediction and it's wrong, update your weights proportionally to the error. It's one of the oldest ideas in machine learning.

For 60 years it sat unused in the deep context of backpropagation-dominated ML. Then in 2021, researchers at IDSIA showed you could apply the delta rule inside a linear attention layer to get **associative memory** — keys map to values, and you can look them up and update them in place. They called it DeltaNet.

The problem: DeltaNet had no forgetting. Old information just accumulated. It needed gating.

That's where Mamba (2023) came in. Mamba showed that learned gating — where the model controls how much information to retain at each step — was powerful and efficient. In late 2024, GatedDeltaNet combined the two: delta rule associative memory with Mamba-style forget gates.

The result is a fixed-size state matrix `S` of shape `(heads, value_dim, key_dim)`. At each token:
1. **Read**: `Sk = S @ k` — retrieve the current best guess for this key
2. **Correct**: `delta = v - Sk` — compute the error
3. **Write**: `S = alpha * S + beta * outer(delta, k)` — update with the correction, where `alpha` (forget gate) and `beta` (write scale) are learned per token

Whether you've processed 10 tokens or 10,000, `S` stays the same size. This is the fundamental efficiency win.

---

## The Port

Porting OLMo Hybrid to MLX required implementing GDN from scratch. No reference implementation exists in MLX. The PyTorch version uses custom CUDA kernels optimized for parallel execution — none of which translate directly.

A few things that made this harder than expected:

**ShortConv.** Before the GDN recurrence, queries, keys, and values pass through a causal depthwise 1D convolution (`kernel_size=4`). MLX doesn't expose grouped convolutions, so I implemented this as a manual sliding-window multiply — selecting the right previous tokens explicitly.

**Hybrid state management.** The model has two fundamentally different kinds of memory: GDN layers carry a recurrent state `(S, conv_context)`, while attention layers carry a standard growing KV cache. A single generation step needs to maintain both simultaneously.

**Silent weight mismatches.** Two bugs took days to track down:
- *QK norm shape*: the norm weight should be `(128,)` per head, not `(3840,)` applied to the full flattened projection. The wrong tensor loaded without error — it just produced wrong outputs.
- *RoPE theta*: the model uses `theta=500,000` for long-context position encoding. My config had `10,000` — the GPT-2 era default. Wrong position encodings cause coherence collapse at longer sequences.

After fixing these, inference worked: 16.3 tok/s prefill, 4.9 tok/s decode on an M4 Max. The decode speed reflects GDN's fundamental nature — each token requires a sequential matrix update that can't be parallelized.

---

## The Fine-Tuning Challenge

A base language model completes text. It doesn't follow instructions. Ask OLMo Hybrid a question and it answers in multiple-choice format, because that's what dominated its training data. To make it useful, you need instruction fine-tuning.

The standard approach for running this on consumer hardware is **LoRA** (Low-Rank Adaptation): freeze the entire base model, then inject small trainable adapter matrices on top of the attention projections. Instead of training 7.4 billion parameters, you train 1.5 million. The frozen base provides the model's knowledge; the adapters teach it how to respond.

For OLMo Hybrid specifically, this meant:
- LoRA adapters on attention `q_proj`, `v_proj`, `o_proj` (the 8 attention layers)
- Full `stop_gradient` on GDN layers — no backprop through the sequential recurrence
- Train on Stanford Alpaca (52K instruction examples)

In theory this should fit in 36GB easily. In practice, it took three separate debugging sessions to get there.

---

## Debugging Session 1: GPU 0%, OOM Before Forward Pass

First run. GPU activity: zero percent. Process killed at 32GB/36GB before a single forward pass.

Something was allocating 32GB just from setup. After some investigation: **AdamW optimizer state**.

Without `model.freeze()`, MLX treats every parameter as trainable. AdamW maintains two moment tensors per trainable parameter for its momentum estimates. That's:

```
7.4 billion params × 2 moment tensors × 2 bytes (float16) = ~59GB
```

The optimizer was trying to allocate ~59GB before computing anything. It killed the process during initialization.

Fix: call `model.freeze()` before injecting LoRA adapters. After this, GPU activates, training starts — but still OOMs.

---

## Debugging Session 2: Trainable Params = 7,432 Million (Should Be 1.5 Million)

After fixing the optimizer state, I printed the trainable parameter count: **7,432.3 million**. The entire model was still trainable.

The issue is subtle. MLX's `model.freeze()` marks parameters frozen at the exact moment it's called. But `inject_lora()` runs *after* `freeze()` — it creates new `mx.array` objects for `lora_a` and `lora_b` and attaches them to the module. These new arrays have no connection to the freeze state. They're born trainable by default, but so is everything else because the parent module's frozen state didn't transfer.

The fix requires an explicit loop after injection:

```python
for layer in model.layers:
    if layer.layer_type == "full_attention":
        for proj_name in ("q_proj", "v_proj", "o_proj"):
            proj = getattr(layer.mixer, proj_name, None)
            if isinstance(proj, LoRALinear):
                proj.unfreeze()      # lora_a + lora_b become trainable
                proj.base.freeze()   # base Linear stays frozen
```

After this: 1,474.6K trainable params. Confirmed correct. But the backward pass still OOMs — now at a different point.

---

## Debugging Session 3: The Memory Wall

With the correct trainable count, the forward pass is fine. A diagnostic probe showed:

```
Weights loaded:      active=14.86GB  cache=0.00GB
After forward T=64:  active=16.44GB  cache=1.68GB
After clear_cache:   active=14.87GB  cache=0.00GB
```

The forward pass adds only 1.58GB of overhead. That clears completely. So far, great.

But the backward pass — `mx.eval(loss, grads)` — gets the process killed immediately. No useful error message; the kernel just terminates.

The math: 14.86GB weights + ~15GB backward overhead ≈ 30GB. With tokenizer, dataset, Python runtime, and GPU driver overhead, that's over 36GB.

Why 15GB for the backward pass on a model with 1.5M trainable params? Because **MLX's autograd tape holds intermediate activations from the entire forward pass until the backward is complete**. Every layer's residual stream, MLP intermediate values, attention matrices — all of it sits in memory simultaneously while gradients are being computed. Activation checkpointing (`mx.checkpoint`) helps at the layer level, but the base model is still 14.86GB and the gradient machinery needs room to work.

The only real solution: **shrink the model itself**.

---

## QLoRA: The Fix

QLoRA is the standard approach to fine-tuning large models on consumer hardware. The idea: quantize the frozen base model to 4-bit integers. The LoRA adapters stay in float16.

In MLX, this is one line:

```python
nn.quantize(model, bits=4, group_size=64)
```

This converts every `nn.Linear` layer from float16 to 4-bit quantized format, storing weights as packed integers plus per-group scale factors. The math:

```
7.4B params × 0.5 bytes (4-bit) ≈ 3.7GB
Was: 14.86GB → saves ~11GB
```

The base weights dequantize on-the-fly during each forward pass, adding a small compute overhead but keeping memory usage tiny. The LoRA adapters (`lora_a`, `lora_b`) stay in float16 and accumulate gradients normally — their math is exact.

After quantization, another freeze/unfreeze cycle is needed because `nn.quantize` replaces `nn.Linear` with `nn.QuantizedLinear`, creating new parameter tensors:

```python
model.freeze()  # freeze the new quantized weight tensors
# then re-unfreeze LoRA params as before
```

---

## Results

```
Weights loaded:      active=4.19GB  cache=14.86GB
After forward T=64:  active=4.27GB  cache=15.14GB
After clear_cache:   active=4.19GB  cache=0.00GB

micro-step 0 backward OK — loss=0.7903  [active=4.20GB  cache=0.00GB]
step     1 | loss 2.5615 | 34 tok/s | 15s | active=4.21GB
step     2 | loss 1.7774 | 31 tok/s | 33s | active=4.21GB
...
```

Active memory during training: **4.21GB**. Stable. Not going up. The `cache=14.86GB` on the first line is the old float16 tensors sitting in MLX's buffer pool — they clear on the first `mx.clear_cache()` and never come back.

The training speed (~20 tok/s at `max_length=128`) reflects GDN's sequential nature: each token requires a recurrent matrix update that can't be parallelized. This is fundamental to the architecture, not an MLX limitation. Fine-tuning the full 52K Alpaca dataset would take about 3 days; a 5K sample fine-tune for instruction following takes around 7 hours.

---

## Why This Is Interesting

OLMo Hybrid 7B is genuinely new territory. GDN hybrid architectures are months old. The LoRA fine-tuning recipe for this architecture didn't exist. There's no QLoRA reference for GDN. The open questions — does LoRA on attention-only work when 75% of the layers are frozen recurrence? does fine-tuning via attention adapters alone teach instruction following effectively? — are still being answered.

The answer to "does it train?" is now yes. The loss curves on 100-sample tests are noisy but training is stable, gradients are flowing, and the adapter weights are updating correctly.

The more interesting questions about whether the resulting model actually follows instructions coherently — those will take a full fine-tuning run and some careful evaluation. That's next.

---

## What's Next

- **Full Alpaca fine-tune** (~7 hours at 5K samples) and qualitative evaluation
- **DPO fine-tuning** (`train_dpo.py`) for preference alignment
- **Longer sequences** — the architecture was designed for 65k tokens; current training at 128-256 barely exercises the GDN's long-context advantage
- **Inference benchmarks** at longer context windows to validate the fixed-memory claim in practice

The repo is public: weights, convert script, generate script, and training code all included.

---

*The model: [allenai/OLMo-Hybrid-7B](https://huggingface.co/allenai/OLMo-Hybrid-7B). The framework: [ml-explore/mlx](https://github.com/ml-explore/mlx). The hardware: Apple M4 Max, 36GB unified memory.*
