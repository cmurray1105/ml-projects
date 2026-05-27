"""
debug_scale.py  —  Hidden state scale + lm_head alignment diagnostics

What this diagnoses:
  1. L2 norm of hidden state at every layer (raw, before model.norm)
  2. Max logit after model.norm + lm_head
  3. Cosine similarity between final normed hidden state and ' Paris' lm_head row
  4. lm_head row norm distribution (sample 1000 rows)

Run:
    python debug_scale.py --weights ./weights/weights.npz
"""

import argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from model import OLMoHybridConfig, OLMoHybrid


def load_weights(model, weights_path):
    data = np.load(weights_path)
    weights = {k: mx.array(data[k]) for k in data.files}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--hf-model", default="allenai/Olmo-Hybrid-7B")
    args = parser.parse_args()

    cfg = OLMoHybridConfig()
    model = OLMoHybrid(cfg)
    load_weights(model, args.weights)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        ids = tok.encode("The capital of France is")
        PARIS_ID = 12366
    except Exception:
        print("Tokenizer unavailable — using fixed ids")
        ids = [791, 6864, 315, 9822, 374]   # approximate
        PARIS_ID = 12366

    input_ids = mx.array([ids], dtype=mx.int32)
    B, T = input_ids.shape

    # ── Setup ──────────────────────────────────────────────────────────────────
    x = model.embed_tokens(input_ids)
    cos, sin = model.rotary(T, offset=0)
    mask = mx.triu(mx.full((T, T), float("-inf")), k=1)[None, None] if T > 1 else None
    gdn_states = [None] * cfg.num_hidden_layers
    kv_caches  = [None] * cfg.num_hidden_layers

    # ── Check lm_head weight row norms (sample) ────────────────────────────────
    lmh = np.array(model.lm_head.weight, dtype=np.float32)   # (vocab, hidden)
    sample_idx = np.random.RandomState(42).choice(len(lmh), 1000, replace=False)
    sample_norms = np.linalg.norm(lmh[sample_idx], axis=1)
    paris_norm = np.linalg.norm(lmh[PARIS_ID])

    print("── lm_head weight diagnostics ────────────────────────────────────────")
    print(f"  shape         : {lmh.shape}")
    print(f"  row norm (1k sample): mean={sample_norms.mean():.4f}  std={sample_norms.std():.4f}  min={sample_norms.min():.4f}  max={sample_norms.max():.4f}")
    print(f"  ' Paris' row norm   : {paris_norm:.4f}")
    print()

    # ── embed_tokens weight norms ──────────────────────────────────────────────
    emb = np.array(model.embed_tokens.weight, dtype=np.float32)
    emb_norms = np.linalg.norm(emb[sample_idx], axis=1)
    print(f"  embed_tokens row norm (1k): mean={emb_norms.mean():.4f}  std={emb_norms.std():.4f}")
    print(f"  lm_head == embed_tokens   : {np.allclose(lmh, emb)}")
    print()
    del lmh, emb   # free memory

    # ── Layer-by-layer hidden state scale ──────────────────────────────────────
    print("── Hidden state L2 norm per layer ────────────────────────────────────")
    print(f"  (last token position, averaged across last-token vector)")
    print()
    print(f"  {'Layer':<14} {'raw_norm':>10}  {'normed_norm':>12}  {'top_logit':>10}  {'paris_logit':>12}")
    print(f"  {'-'*14} {'-'*10}  {'-'*12}  {'-'*10}  {'-'*12}")

    def check(x, label):
        mx.eval(x)
        x_np = np.array(x[0, -1, :], dtype=np.float32)   # last token, (hidden,)

        # Raw norm
        raw_norm = float(np.linalg.norm(x_np))

        # After final model norm
        xn = model.norm(x[0:1, -1:, :])   # (1, 1, hidden)
        mx.eval(xn)
        xn_np = np.array(xn[0, 0, :], dtype=np.float32)
        normed_norm = float(np.linalg.norm(xn_np))

        # lm_head projection
        lg = model.lm_head(xn)
        mx.eval(lg)
        lg_np = np.array(lg[0, 0, :], dtype=np.float32)

        top_logit  = float(lg_np.max())
        paris_logit = float(lg_np[PARIS_ID])

        print(f"  {label:<14} {raw_norm:>10.3f}  {normed_norm:>12.4f}  {top_logit:>10.3f}  {paris_logit:>12.3f}")

    check(x, "embed-only")

    for i, layer in enumerate(model.layers):
        x, gdn_states[i], kv_caches[i] = layer(
            x, cos=cos, sin=sin, mask=mask,
            gdn_state=gdn_states[i], kv_cache=kv_caches[i],
        )
        ltype = "GDN" if cfg.layer_type(i) == "linear_attention" else "ATN"
        if i < 4 or (i + 1) % 4 == 0 or i >= 28:
            check(x, f"L{i:>2} [{ltype}]")

    # ── Final cosine similarity with Paris ────────────────────────────────────
    print()
    print("── Final hidden state cosine sim with ' Paris' lm_head row ──────────")
    xn_final = model.norm(x[0:1, -1:, :])
    mx.eval(xn_final)
    xn_np = np.array(xn_final[0, 0, :], dtype=np.float32)

    lmh_fresh = np.array(model.lm_head.weight, dtype=np.float32)
    paris_vec = lmh_fresh[PARIS_ID]
    top20_idx = np.argsort(-lmh_fresh @ xn_np)[:5]

    cos_paris = float(np.dot(xn_np, paris_vec) / (np.linalg.norm(xn_np) * np.linalg.norm(paris_vec) + 1e-8))
    print(f"  cosine( hidden_final, lm_head[Paris] ) = {cos_paris:.6f}")
    print(f"  raw dot product = {float(np.dot(xn_np, paris_vec)):.4f}")
    print(f"  hidden norm = {np.linalg.norm(xn_np):.4f}  paris_vec norm = {np.linalg.norm(paris_vec):.4f}")
    print()
    print("  Top-5 tokens by cosine sim with final hidden state:")
    try:
        from transformers import AutoTokenizer
        tok2 = AutoTokenizer.from_pretrained(args.hf_model)
        for rank, idx in enumerate(top20_idx):
            print(f"    #{rank+1}: id={idx}  '{tok2.decode([int(idx)])}'  logit={float(lmh_fresh[idx] @ xn_np):.3f}")
    except Exception:
        for rank, idx in enumerate(top20_idx):
            print(f"    #{rank+1}: id={idx}  logit={float(lmh_fresh[idx] @ xn_np):.3f}")

    print()
    print("── Summary: likely culprit ───────────────────────────────────────────")
    final_top = float((lmh_fresh @ xn_np).max())
    print(f"  Expected max logit for healthy 7B : ~15-25")
    print(f"  Actual max logit                  : {final_top:.3f}")
    if final_top < 5:
        if np.linalg.norm(xn_np) < 0.5:
            print("  → DIAGNOSIS: Final hidden state near-zero after model.norm (scale collapse)")
        elif np.array([np.linalg.norm(lmh_fresh[i]) for i in sample_idx]).mean() < 0.1:
            print("  → DIAGNOSIS: lm_head weight rows near-zero (weight loading failure)")
        else:
            print("  → DIAGNOSIS: Hidden state direction is wrong (orthogonal to all vocab rows)")
            print("               Model is computing something but in the wrong subspace.")
    del lmh_fresh


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
