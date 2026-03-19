"""Download Python wheels for offline MCP server deployment.

Creates a deps.zip containing wheel files that can be uploaded to
H2OGPTe alongside the MCP server. The agent prompt template instructs
the agent to unzip and pip install from these wheels.

Packages already pre-installed in the H2OGPTe runtime are excluded
to keep the zip small.

Usage:
    # Download wheels for current platform
    python build_deps.py

    # Cross-platform (e.g. for Linux deployment from macOS)
    python build_deps.py --platform manylinux2014_x86_64 --python-version 3.11

    # Include NLTK data (punkt_tab, wordnet) for METEOR metric
    python build_deps.py --include-nltk-data

    # Custom output name
    python build_deps.py -o my_deps.zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# Packages already available in the H2OGPTe runtime — do not bundle.
EXCLUDE_WHEELS = {
    # Deep Learning / Inference
    "torch", "torchvision", "transformers", "accelerate",
    "numpy", "pandas", "scikit-learn", "scipy",
    # Data / Arrow
    "pyarrow",
    # Web / API / HTTP
    "requests", "httpx", "httpcore", "urllib3", "certifi",
    "charset-normalizer", "idna", "h11", "anyio", "sniffio",
    # Pydantic / typing
    "pydantic", "pydantic-core", "pydantic-settings",
    "annotated-types", "typing-extensions", "typing-inspection",
    # UI / plotting (not needed for MCP)
    "streamlit", "matplotlib", "pillow", "pydeck",
    "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing",
    # Torch transitive deps
    "sympy", "mpmath", "networkx",
    # Utilities already in H2OGPTe
    "tqdm", "rich", "pygments", "markdown-it-py", "mdurl",
    "openpyxl", "et-xmlfile",
    "pyyaml", "packaging", "filelock", "safetensors",
    "setuptools", "six",
    # Jinja / templating
    "jinja2", "markupsafe",
    # HuggingFace ecosystem (hub, tokenizers already via transformers)
    "huggingface-hub", "hf-xet", "tokenizers",
    # Async / aiohttp
    "aiohttp", "aiosignal", "aiohappyeyeballs", "frozenlist",
    "multidict", "yarl", "propcache", "attrs",
    # Job / threading
    "joblib", "threadpoolctl",
    # Other pre-installed
    "click", "wrapt", "jsonschema", "jsonschema-specifications",
    "referencing", "rpds-py", "narwhals",
}


def download_wheels(
    requirements_path: Path,
    dest_dir: Path,
    platform: str | None = None,
    python_version: str | None = None,
) -> None:
    """Download wheels and remove pre-installed packages."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(requirements_path),
        "-d", str(dest_dir),
        "--only-binary", ":all:",
    ]
    if platform:
        cmd += ["--platform", platform]
    if python_version:
        cmd += ["--python-version", python_version]
    if platform or python_version:
        cmd += ["--no-deps"]

    print(f"Downloading wheels...")
    print(f"  Requirements: {requirements_path}")
    print(f"  Platform:     {platform or 'current'}")
    print(f"  Python:       {python_version or f'{sys.version_info.major}.{sys.version_info.minor}'}")
    print(f"  Dest:         {dest_dir}")

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        # Two-pass approach: first download what we can as wheels-only
        # (one package at a time), then retry failures without --only-binary
        # so pure-Python sdists (e.g. rouge-score) are allowed.
        print("  Some packages lack wheels — falling back to per-package download...")
        with open(requirements_path) as f:
            lines = f.readlines()

        failed_pkgs = []
        for line in lines:
            # Strip inline comments (e.g. "pandas==2.2.0  # Data handling")
            line = line.split("#")[0].strip()
            if not line:
                continue
            # Try wheels-only first for each package
            single_cmd = [
                sys.executable, "-m", "pip", "download",
                line, "-d", str(dest_dir), "--only-binary", ":all:",
            ]
            if platform:
                single_cmd += ["--platform", platform]
            if python_version:
                single_cmd += ["--python-version", python_version]
            if platform or python_version:
                single_cmd += ["--no-deps"]
            result = subprocess.run(single_cmd, capture_output=True)
            if result.returncode != 0:
                failed_pkgs.append(line)

        # Retry failures without --only-binary (allows pure-Python sdists)
        if failed_pkgs:
            print(f"  Retrying {len(failed_pkgs)} packages without --only-binary: "
                  f"{[p.split('>=')[0].split('==')[0] for p in failed_pkgs]}")
            for pkg in failed_pkgs:
                single_cmd = [
                    sys.executable, "-m", "pip", "download",
                    pkg, "-d", str(dest_dir),
                ]
                if platform:
                    single_cmd += ["--platform", platform]
                if python_version:
                    single_cmd += ["--python-version", python_version]
                if platform or python_version:
                    single_cmd += ["--no-deps"]
                subprocess.check_call(single_cmd)

    # Convert any .tar.gz sdists to wheels (sandbox may lack build tools)
    sdists = list(dest_dir.glob("*.tar.gz"))
    if sdists:
        print(f"  Converting {len(sdists)} sdist(s) to wheels...")
        for sdist in sdists:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", str(sdist),
                 "--no-deps", "-w", str(dest_dir)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                sdist.unlink()
                print(f"    Converted: {sdist.name} -> wheel")
            else:
                print(f"    Warning: Could not convert {sdist.name}, keeping sdist")

    # Remove pre-installed packages
    removed = []
    for whl in list(dest_dir.glob("*.whl")) + list(dest_dir.glob("*.tar.gz")):
        pkg_name = whl.name.split("-")[0].lower().replace("_", "-")
        if pkg_name in EXCLUDE_WHEELS:
            removed.append((whl.name, whl.stat().st_size))
            whl.unlink()

    if removed:
        saved = sum(s for _, s in removed) / (1024 * 1024)
        print(f"  Removed {len(removed)} pre-installed packages ({saved:.1f} MB saved):")
        for name, _ in removed:
            print(f"    - {name}")

    remaining = list(dest_dir.glob("*.whl")) + list(dest_dir.glob("*.tar.gz"))
    print(f"  Final package count: {len(remaining)}")

    # Write manifest
    manifest = {
        "platform": platform or "current",
        "python_version": python_version or f"{sys.version_info.major}.{sys.version_info.minor}",
        "bundled_at": datetime.now(timezone.utc).isoformat(),
        "package_count": len(remaining),
    }
    manifest_path = dest_dir / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path}")


def download_nltk_data(dest_dir: Path) -> None:
    """Download NLTK data required for METEOR metric.

    METEOR needs: punkt_tab (tokenizer), wordnet (synonyms),
    omw-1.4 (Open Multilingual Wordnet, required by wordnet),
    and punkt (legacy tokenizer fallback).

    NLTK downloads zip files but doesn't always extract them (Zip Slip
    protection blocks some).  nltk.data.find() needs extracted directories
    for corpora, so we manually extract all zip files after downloading.
    """
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NLTK data to {dest_dir}...")

    import nltk
    for pkg in ("punkt_tab", "punkt", "wordnet", "omw-1.4"):
        nltk.download(pkg, download_dir=str(dest_dir), quiet=True)
        print(f"  Downloaded: {pkg}")

    # Extract any zip files that NLTK left unextracted (Zip Slip protection
    # blocks automatic extraction for some packages).  nltk.data.find()
    # requires extracted directories, not zip files.
    for subdir in ("corpora", "tokenizers"):
        data_subdir = dest_dir / subdir
        if not data_subdir.is_dir():
            continue
        for zf in data_subdir.glob("*.zip"):
            extracted_name = zf.stem  # e.g. "wordnet" from "wordnet.zip"
            extracted_dir = data_subdir / extracted_name
            if not extracted_dir.is_dir():
                try:
                    with zipfile.ZipFile(zf, "r") as z:
                        z.extractall(data_subdir)
                    print(f"  Extracted: {subdir}/{zf.name}")
                except Exception as e:
                    print(f"  Warning: Could not extract {zf.name}: {e}")


def build_deps_zip(
    output: str = "deps.zip",
    platform: str | None = None,
    python_version: str | None = None,
    include_nltk_data: bool = False,
) -> None:
    """Build deps.zip with wheels and optional NLTK data."""
    server_dir = Path(__file__).parent
    requirements_path = server_dir.parent / "requirements-mcp.txt"
    if not requirements_path.exists():
        requirements_path = server_dir.parent / "requirements.txt"

    # Also copy requirements.txt into the zip for the agent to reference
    tmp_dir = server_dir / "dist_deps"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    deps_dir = tmp_dir / "deps"
    download_wheels(requirements_path, deps_dir, platform, python_version)

    # Copy requirements.txt
    shutil.copy2(requirements_path, tmp_dir / "requirements.txt")
    print(f"  Copied {requirements_path.name} -> dist_deps/requirements.txt")

    # NLTK data
    if include_nltk_data:
        nltk_dir = tmp_dir / "nltk_data"
        download_nltk_data(nltk_dir)

    # Zip everything
    zip_path = server_dir / output
    print(f"\nCreating {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(tmp_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(tmp_dir)
                zf.write(file_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Created: {zip_path} ({size_mb:.1f} MB)")

    # Cleanup
    shutil.rmtree(tmp_dir)
    print("  Cleaned up dist_deps/")


def main():
    parser = argparse.ArgumentParser(
        description="Build deps.zip with Python wheels for offline MCP deployment"
    )
    parser.add_argument(
        "-o", "--output",
        default="deps.zip",
        help="Output zip filename (default: deps.zip)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Target platform (e.g. manylinux2014_x86_64, macosx_11_0_arm64)",
    )
    parser.add_argument(
        "--python-version",
        default=None,
        help="Target Python version (e.g. 3.11, 3.12, 3.13)",
    )
    parser.add_argument(
        "--include-nltk-data",
        action="store_true",
        help="Bundle NLTK data (punkt_tab, wordnet) for METEOR metric",
    )
    args = parser.parse_args()

    build_deps_zip(
        output=args.output,
        platform=args.platform,
        python_version=args.python_version,
        include_nltk_data=args.include_nltk_data,
    )


if __name__ == "__main__":
    main()
