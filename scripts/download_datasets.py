#!/usr/bin/env python3
"""Simple dataset downloader based on datasets_manifest.yaml.

Usage:
  python scripts/download_datasets.py [--manifest path] [--list] [--download <id>]

Behavior:
 - For entries with method 'direct' it will download the given URL.
 - For entries with method 'hf' it will attempt to use the `datasets` library to download.
 - For entries with method 'manual' it prints instructions.

This script avoids adding any files to git; it's a helper to reproduce dataset layout.
"""
import argparse
import os
import sys
import yaml
import shutil
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def load_manifest(path: Path):
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path):
    if requests is None:
        print("requests not installed; cannot download", url)
        return False
    ensure_dir(dest)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return True


def do_hf_download(hf_name: str, target_dir: Path):
    if load_dataset is None:
        print("datasets library not installed; install with `pip install datasets` to auto-download HF datasets")
        return False
    print(f"Attempting to download HuggingFace dataset '{hf_name}' into {target_dir}")
    ds = load_dataset(hf_name)
    # Save minimal local cache info: write dataset reproducible note
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / "README.hf.txt", "w", encoding="utf8") as f:
        f.write(f"Downloaded dataset: {hf_name}\n")
    print("HF dataset fetched (check local cache under HF cache / datasets library)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="datasets_manifest.yaml")
    p.add_argument("--list", action="store_true")
    p.add_argument("--download", help="download dataset by id (from manifest)")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("manifest not found:", manifest_path)
        sys.exit(2)

    manifest = load_manifest(manifest_path)
    ds_list = manifest.get("datasets", [])

    if args.list:
        for d in ds_list:
            print(f"{d.get('id')}: {d.get('name')} -> {d.get('path')} (method={d.get('method')})")
        return

    to_process = ds_list
    if args.download:
        to_process = [d for d in ds_list if d.get("id") == args.download]
        if not to_process:
            print("No manifest entry for id", args.download)
            return

    for d in to_process:
        mid = d.get("id")
        path = Path(d.get("path"))
        method = d.get("method")
        print("-" * 60)
        print(f"Processing: {mid} -> {path} (method={method})")
        if method == "manual":
            print("Manual instructions:")
            print(d.get("instruction"))
            continue

        if method == "hf":
            hf_name = d.get("hf_name")
            if hf_name:
                do_hf_download(hf_name, path)
            else:
                print("hf entry missing 'hf_name', skipping")
            continue

        if method == "direct":
            url = d.get("url")
            if not url:
                print("direct entry missing url, skipping")
                continue
            dest = path
            if dest.is_dir() or str(dest).endswith("/"):
                dest = Path(path) / Path(url).name
            download_file(url, dest)
            continue

        print("Unknown method", method)


if __name__ == "__main__":
    main()
