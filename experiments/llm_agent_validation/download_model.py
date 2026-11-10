#!/usr/bin/env python3
"""Download public official weights at the exact frozen revision, anonymously."""
from pathlib import Path
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    local_dir=HERE / "models" / "Qwen2.5-3B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
    max_workers=2, token=False,
)
