"""
Download all three model safetensors to models/hf_cache/
Run from project root: python scripts/download_models.py

Requires HF login for Gemma (one-time):
    huggingface-cli login
    Then accept licence at https://huggingface.co/google/gemma-3-4b-it

HF_HOME must point to ~/Research/models/hf_cache (set in .bashrc).
"""

import os
import sys
from huggingface_hub import snapshot_download

MODELS = [
    {
        "name":      "Qwen-MediCare-BD",
        "repo_id":   "CBrootA/Qwen-MediCare-BD",
        "local_dir": "models/hf_cache/qwen_medicare_bd",
        "ignore":    ["*.bin", "*.pt", "*.gguf"],
        "note":      None,
    },
    {
        "name":      "Gemma-3-4B-it",
        "repo_id":   "google/gemma-3-4b-it",
        "local_dir": "models/hf_cache/gemma_3_4b_it",
        "ignore":    ["*.bin"],
        "note":      "Requires: huggingface-cli login AND licence accepted at "
                     "https://huggingface.co/google/gemma-3-4b-it",
    },
    {
        "name":      "Phi-4-mini-instruct",
        "repo_id":   "microsoft/Phi-4-mini-instruct",
        "local_dir": "models/hf_cache/phi4_mini_instruct",
        "ignore":    ["*.bin", "*.pt"],
        "note":      None,
    },
]


def download(model: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"Downloading: {model['name']}")
    print(f"  repo  : {model['repo_id']}")
    print(f"  dest  : {model['local_dir']}")
    if model["note"]:
        print(f"  NOTE  : {model['note']}")
    print(f"{'='*60}")

    try:
        path = snapshot_download(
            repo_id=model["repo_id"],
            local_dir=model["local_dir"],
            ignore_patterns=model["ignore"],
        )
        files = os.listdir(path)
        safetensors = [f for f in files if f.endswith(".safetensors")]
        print(f"  Done. {len(safetensors)} safetensor shard(s) in {path}")
        print(f"  Files: {files}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working dir : {os.getcwd()}")
    print(f"HF_HOME     : {os.environ.get('HF_HOME', '(not set — using default ~/.cache)')}")

    results = {}
    for model in MODELS:
        results[model["name"]] = download(model)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    if not all(results.values()):
        sys.exit(1)
