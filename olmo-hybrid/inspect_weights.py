"""
inspect_weights.py

Run this FIRST to see the exact weight names in the HF checkpoint.
It downloads only the config + model.safetensors.index.json (tiny),
then optionally lists all tensor names so we can write the converter.

Usage:
    pip install huggingface_hub safetensors transformers --break-system-packages
    python inspect_weights.py

    # To also download the full weights (≈14GB):
    python inspect_weights.py --download
"""

import argparse
import json
from pathlib import Path

MODEL_ID = "allenai/Olmo-Hybrid-7B"


def inspect_config():
    from huggingface_hub import hf_hub_download
    import json

    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    print("=== CONFIG ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    return cfg


def inspect_weight_index():
    """Download only the index file (a few KB) to list all tensor names."""
    from huggingface_hub import hf_hub_download

    try:
        idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)

        tensors = sorted(idx["weight_map"].keys())
        print(f"\n=== TENSOR NAMES ({len(tensors)} total) ===")
        for name in tensors:
            print(f"  {name}")
        return tensors
    except Exception as e:
        print(f"No index file (model might be single-shard): {e}")
        return None


def inspect_single_shard():
    """If single shard, download and list names."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(MODEL_ID, "model.safetensors")
    with safe_open(path, framework="numpy") as f:
        names = sorted(f.keys())

    print(f"\n=== TENSOR NAMES ({len(names)} total) ===")
    for name in names:
        print(f"  {name}")
    return names


def download_full_model():
    from huggingface_hub import snapshot_download
    local = snapshot_download(MODEL_ID, ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
    print(f"\nDownloaded to: {local}")
    return local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download full model weights")
    args = parser.parse_args()

    cfg = inspect_config()
    tensors = inspect_weight_index()
    if tensors is None:
        tensors = inspect_single_shard()

    if args.download:
        download_full_model()

    # Print a summary of layer structure for the converter
    if tensors:
        print("\n=== LAYER STRUCTURE SUMMARY ===")
        # Find unique prefixes up to layer index
        layer_tensors = [t for t in tensors if "layers." in t]
        if layer_tensors:
            # Group by layer number
            from collections import defaultdict
            by_layer = defaultdict(list)
            for t in layer_tensors:
                parts = t.split(".")
                try:
                    idx = parts.index("layers")
                    layer_num = int(parts[idx + 1])
                    suffix = ".".join(parts[idx + 2:])
                    by_layer[layer_num].append(suffix)
                except (ValueError, IndexError):
                    pass

            # Print layer 0 and layer 3 (should be GDN and Attention respectively)
            for layer_idx in [0, 1, 3]:
                if layer_idx in by_layer:
                    print(f"\nLayer {layer_idx} tensors:")
                    for t in sorted(by_layer[layer_idx]):
                        print(f"    {t}")
