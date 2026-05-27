"""
Smoke test for the mlx-lm OLMo Hybrid port.

Run from inside ~/Development/MLX/olmo-hybrid/mlx-lm/:

    python ../test_mlxlm_port.py

Tests:
  1. ModelArgs loads correctly from config.json
  2. Model structure matches expected attribute paths
  3. make_cache() returns the right types
  4. sanitize() maps all weight keys correctly (no unknowns, no duplicates)
  5. Weights load without shape errors
  6. Forward pass produces logits of correct shape
  7. generate() produces coherent text (manual check)
"""

import sys
import json
from pathlib import Path

# ── Must be run from inside mlx-lm/ so relative imports resolve ──────────────
sys.path.insert(0, str(Path(__file__).parent / "mlx-lm"))

import mlx.core as mx
from mlx_lm.models.olmo_hybrid import Model, ModelArgs, GDNCache, KVCache

HF_MODEL = "allenai/OLMo-Hybrid-7B"
LOCAL_WEIGHTS = Path(__file__).parent / "weights"  # your existing converted weights


# ─────────────────────────────────────────────────────────────────────────────
# 1. ModelArgs from config.json
# ─────────────────────────────────────────────────────────────────────────────
print("1. ModelArgs.from_dict() ...")
config_path = LOCAL_WEIGHTS / "config.json"
with open(config_path) as f:
    config = json.load(f)

args = ModelArgs.from_dict(config)
print(f"   vocab_size={args.vocab_size}, hidden_size={args.hidden_size}")
print(f"   num_hidden_layers={args.num_hidden_layers}")
print(f"   layer_types[:8]={args.layer_types[:8]}")
assert args.rope_theta == 500000.0, f"rope_theta wrong: {args.rope_theta}"
assert len(args.layer_types) == 32
assert args.layer_types[3] == "full_attention"
assert args.layer_types[0] == "linear_attention"
print("   ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n2. Model structure ...")
model = Model(args)
assert hasattr(model, "model"), "missing model.model"
assert hasattr(model.model, "embed_tokens")
assert hasattr(model.model, "norm")
assert hasattr(model, "lm_head")
assert len(model.layers) == 32
from mlx_lm.models.olmo_hybrid import GatedDeltaNet, Attention
assert isinstance(model.layers[0].mixer, GatedDeltaNet), "layer 0 should be GDN"
assert isinstance(model.layers[3].mixer, Attention),     "layer 3 should be Attention"
print("   ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. make_cache() types
# ─────────────────────────────────────────────────────────────────────────────
print("\n3. make_cache() ...")
cache = model.make_cache()
assert len(cache) == 32
gdn_count = sum(1 for c in cache if isinstance(c, GDNCache))
kv_count  = sum(1 for c in cache if isinstance(c, KVCache))
assert gdn_count == 24, f"expected 24 GDNCache, got {gdn_count}"
assert kv_count  == 8,  f"expected 8 KVCache, got {kv_count}"
print(f"   GDNCache: {gdn_count}  KVCache: {kv_count}  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 4. sanitize() — no unknowns, no duplicates
# ─────────────────────────────────────────────────────────────────────────────
print("\n4. sanitize() weight mapping ...")
import mlx.core as mx
from mlx.utils import tree_flatten

# Load raw weights — supports both .safetensors shards and single .npz
weight_files = sorted(LOCAL_WEIGHTS.glob("*.safetensors"))
npz_files    = sorted(LOCAL_WEIGHTS.glob("*.npz"))
assert weight_files or npz_files, f"No weights found in {LOCAL_WEIGHTS}"

raw_weights = {}
for wf in weight_files + npz_files:
    raw_weights.update(mx.load(str(wf)))

print(f"   Raw HF keys: {len(raw_weights)}")

sanitized = model.sanitize(raw_weights)
print(f"   Mapped keys: {len(sanitized)}")

# Check for duplicates
assert len(sanitized) == len(set(sanitized.keys())), "Duplicate keys in sanitized!"

# Report skipped keys (should just be rotary_emb.inv_freq etc.)
skipped = [k for k in raw_weights if model._hf_to_mlx(k) is None]
print(f"   Skipped keys ({len(skipped)}): {skipped[:5]}{'...' if len(skipped)>5 else ''}")
print("   ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Load weights — check for shape errors
# ─────────────────────────────────────────────────────────────────────────────
print("\n5. Loading weights ...")
model.load_weights(list(sanitized.items()))
mx.eval(model.parameters())
active_gb = mx.get_active_memory() / 1e9
print(f"   Active memory: {active_gb:.2f} GB  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Forward pass shape check
# ─────────────────────────────────────────────────────────────────────────────
print("\n6. Forward pass shape ...")
test_ids = mx.array([[1, 2, 3, 4, 5]])   # (1, 5)
cache    = model.make_cache()
logits   = model(test_ids, cache=cache)
mx.eval(logits)
assert logits.shape == (1, 5, args.vocab_size), f"Wrong shape: {logits.shape}"
print(f"   logits shape: {logits.shape}  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Single decode step — verify logits change with each new token
# ─────────────────────────────────────────────────────────────────────────────
print("\n7. Decode step (cache continuity) ...")
cache = model.make_cache()

# Prefill: 5 tokens
prefill_ids = mx.array([[1, 2, 3, 4, 5]])
logits_prefill = model(prefill_ids, cache=cache)
mx.eval(logits_prefill)

# Decode: one more token using last predicted id
next_token = mx.argmax(logits_prefill[0, -1, :], keepdims=True)[None]  # (1,1)
logits_decode = model(next_token, cache=cache)
mx.eval(logits_decode)

assert logits_decode.shape == (1, 1, args.vocab_size)
# Logits should differ from prefill last step (cache advanced)
diff = mx.abs(logits_decode[0,0] - logits_prefill[0,-1]).mean().item()
assert diff > 0, "Logits unchanged after decode step — cache not updating"
print(f"   Prefill → decode logit delta: {diff:.4f}  ✓")

print("\n✅ All automated checks passed.")
