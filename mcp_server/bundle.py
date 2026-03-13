"""
Bundle MCP server and dependencies into a deployable zip file.

Supports optional wheel bundling for airgapped environments:
    python bundle.py --include-wheels --platform manylinux2014_x86_64 --python-version 3.11
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Directories and files to exclude from bundling
EXCLUDE_PATTERNS = {
    '__pycache__',
    '.pyc',
    '.pyo',
    '.git',
    '.DS_Store',
    '.env',
}

# PyTorch index URLs by variant
TORCH_INDEX_URLS = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
}


def should_exclude(path: str) -> bool:
    """Check if a path should be excluded from the bundle."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path:
            return True
    return False


def download_wheels(requirements_path: Path, dest_dir: Path, platform: str = None,
                    python_version: str = None, torch_variant: str = "cpu"):
    """Download wheel files for all dependencies.

    Args:
        requirements_path: Path to requirements.txt.
        dest_dir: Directory to download wheels into.
        platform: Target platform (e.g., manylinux2014_x86_64). If None, uses current platform.
        python_version: Target Python version (e.g., 3.11). If None, uses current interpreter.
        torch_variant: PyTorch variant (cpu, cu118, cu121, cu124).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cross_platform = bool(platform or python_version)

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
    if cross_platform:
        # When cross-downloading, pip requires --implementation and --abi
        abi = f"cp{python_version.replace('.', '')}" if python_version else f"cp{sys.version_info.major}{sys.version_info.minor}"
        cmd += ["--implementation", "cp", "--abi", abi]

    # Add torch index URL
    if torch_variant in TORCH_INDEX_URLS:
        cmd += ["--extra-index-url", TORCH_INDEX_URLS[torch_variant]]

    print(f"  Downloading wheels...")
    print(f"    Platform: {platform or 'current'}")
    print(f"    Python: {python_version or 'current'}")
    print(f"    Torch variant: {torch_variant}")

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        if cross_platform:
            # Cross-platform requires --only-binary :all:, cannot retry without it.
            # Try again without --extra-index-url (use default PyPI for all packages).
            print(f"  Warning: pip download failed (a package may not exist on the torch index).")
            print(f"  Retrying with default PyPI only (no torch-specific index)...")
            torch_url = TORCH_INDEX_URLS.get(torch_variant, "")
            cmd_retry = [c for c in cmd if c not in ("--extra-index-url", torch_url)]
            try:
                subprocess.check_call(cmd_retry)
            except subprocess.CalledProcessError:
                print(f"\n  ERROR: pip download failed for cross-platform target.")
                print(f"  This usually means a package has no wheel for the specified")
                print(f"  platform/python combination. Common fixes:")
                print(f"    - torch>=2.8.0 requires manylinux_2_28 (not manylinux2014)")
                print(f"    - torch>=2.8.0 requires Python >=3.12 (no cp311 wheels)")
                print(f"    - Try: --platform manylinux_2_28_x86_64 --python-version 3.13")
                print(f"    - Or omit --platform/--python-version to use current system")
                raise
        else:
            # Native platform: can retry without --only-binary to allow source dists
            print(f"  Warning: pip download failed with --only-binary :all:.")
            print(f"  Retrying without --only-binary to allow source distributions...")
            cmd_retry = [c for c in cmd if c not in ("--only-binary", ":all:")]
            subprocess.check_call(cmd_retry)

    # Remove packages already available in the H2OGPTe runtime environment.
    # Only evaluation-specific packages (rouge-score, bert-score, spacy, etc.)
    # need to be bundled; everything else is pre-installed.
    EXCLUDE_WHEELS = {
        # Deep Learning / Inference (pre-installed in H2OGPTe)
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
        # HuggingFace ecosystem (hub, tokenizers already in H2OGPTe via transformers)
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
    removed = []
    for whl in list(dest_dir.glob("*.whl")) + list(dest_dir.glob("*.tar.gz")):
        pkg_name = whl.name.split("-")[0].lower().replace("_", "-")
        if pkg_name in EXCLUDE_WHEELS:
            removed.append((whl.name, whl.stat().st_size))
            whl.unlink()
    if removed:
        saved = sum(s for _, s in removed) / (1024 * 1024)
        print(f"  Removed {len(removed)} excluded transitive deps ({saved:.1f} MB saved):")
        for name, _ in removed:
            print(f"    - {name}")

    # Count remaining packages
    wheel_files = list(dest_dir.glob("*.whl")) + list(dest_dir.glob("*.tar.gz"))
    print(f"  Final package count: {len(wheel_files)}")

    # Write manifest
    manifest = {
        "platform": platform or "current",
        "python_version": python_version or f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_variant": torch_variant,
        "bundled_at": datetime.now(timezone.utc).isoformat(),
        "package_count": len(wheel_files),
    }
    manifest_path = dest_dir / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path}")


def download_spacy_model(dest_dir: Path):
    """Download the en_core_web_sm spaCy model wheel."""
    cmd = [
        sys.executable, "-m", "pip", "download",
        "en-core-web-sm",
        "-d", str(dest_dir),
        "--no-deps",
        "--extra-index-url", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    ]

    # spaCy models are typically pure Python (py3-none-any), so platform flags are not needed
    print(f"  Downloading spaCy en_core_web_sm model...")
    try:
        subprocess.check_call(cmd)
        print(f"  spaCy model downloaded.")
    except subprocess.CalledProcessError:
        # Fallback: download directly via pip download with spaCy's find-links
        print(f"  Retrying spaCy model download with find-links...")
        cmd_fallback = [
            sys.executable, "-m", "pip", "download",
            "en-core-web-sm==3.7.1",
            "-d", str(dest_dir),
            "--no-deps",
            "--find-links", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1",
        ]
        subprocess.check_call(cmd_fallback)
        print(f"  spaCy model downloaded (fallback).")


def install_vendored_deps(requirements_path: Path, dest_dir: Path):
    """Install packages directly into a vendor directory (no .whl files).

    This creates a site-packages-like directory that can be added to sys.path
    at runtime, bypassing pip entirely on the target environment.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Packages already in H2OGPTe — skip these
    SKIP_PACKAGES = [
        "torch", "torchvision", "transformers", "accelerate",
        "numpy", "pandas", "scikit-learn", "scipy", "pyarrow",
        "requests", "httpx", "httpcore", "urllib3", "certifi",
        "charset-normalizer", "idna", "h11", "anyio", "sniffio",
        "pydantic", "pydantic-core", "pydantic-settings",
        "annotated-types", "typing-extensions", "typing-inspection",
        "streamlit", "matplotlib", "pillow", "pydeck",
        "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing",
        "sympy", "mpmath", "networkx",
        "tqdm", "rich", "pygments", "markdown-it-py", "mdurl",
        "openpyxl", "et-xmlfile",
        "pyyaml", "packaging", "filelock", "safetensors",
        "setuptools", "six",
        "jinja2", "markupsafe",
        "huggingface-hub", "hf-xet", "tokenizers",
        "aiohttp", "aiosignal", "aiohappyeyeballs", "frozenlist",
        "multidict", "yarl", "propcache", "attrs",
        "joblib", "threadpoolctl",
        "click", "wrapt", "jsonschema", "jsonschema-specifications",
        "referencing", "rpds-py", "narwhals",
    ]

    print(f"  Installing vendored dependencies into {dest_dir}...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_path),
        "--target", str(dest_dir),
        "--no-deps",
    ]
    # First install all direct deps (no transitive) to the target
    subprocess.check_call(cmd)

    # Now install with deps but exclude pre-installed packages
    cmd_with_deps = [
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_path),
        "--target", str(dest_dir),
        "--upgrade",
    ]
    subprocess.check_call(cmd_with_deps)

    # Remove pre-installed packages from vendor dir.
    # Map pip package names to their actual directory names on disk.
    SKIP_DIRS = {
        # Exact directory/file names to remove (case-sensitive on disk)
        "torch", "torchgen", "torchvision", "transformers", "transformers_modules",
        "numpy", "numpy.libs", "pandas", "sklearn", "scipy", "scipy.libs",
        "pyarrow", "requests", "httpx", "httpcore", "urllib3", "certifi",
        "charset_normalizer", "idna", "h11", "anyio", "sniffio",
        "pydantic", "pydantic_core", "pydantic_settings",
        "annotated_types", "typing_extensions", "typing_inspection",
        "streamlit", "matplotlib", "mpl_toolkits", "PIL", "pydeck",
        "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing",
        "sympy", "mpmath", "networkx",
        "tqdm", "rich", "pygments", "markdown_it", "mdurl",
        "openpyxl", "et_xmlfile",
        "yaml", "_yaml", "packaging", "filelock", "safetensors",
        "setuptools", "_distutils_hack", "pkg_resources", "distutils",
        "six", "six.py",
        "jinja2", "markupsafe",
        "huggingface_hub", "hf_xet",
        "tokenizers",
        "aiohttp", "aiosignal", "aiohappyeyeballs", "frozenlist",
        "multidict", "yarl", "propcache", "attr", "attrs",
        "joblib", "threadpoolctl",
        "click", "wrapt", "jsonschema", "jsonschema_specifications",
        "referencing", "rpds", "narwhals",
        # Transitive deps also in H2OGPTe
        "websockets", "toml", "smmap", "gitdb", "git",
        "colorama", "shellingham", "pathspec",
        "bs4", "beautifulsoup4", "soupsieve", "lxml",
    }

    removed = []
    for item in list(dest_dir.iterdir()):
        name = item.name
        # Remove .dist-info dirs for any skipped package
        if name.endswith(".dist-info") or name.endswith(".data"):
            pkg = name.rsplit("-", 2)[0].lower().replace("-", "_")
            normalized_skips = {p.replace("-", "_") for p in SKIP_PACKAGES}
            if pkg in normalized_skips:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
                removed.append(name)
                continue
        # Remove exact directory/file matches (case-insensitive)
        name_lower = name.lower().rstrip(".py")
        if name in SKIP_DIRS or name_lower in {s.lower() for s in SKIP_DIRS}:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed.append(name)

    if removed:
        print(f"  Removed {len(removed)} pre-installed packages from vendor/")

    # Count remaining packages
    pkg_dirs = [d for d in dest_dir.iterdir() if d.is_dir() and not d.name.endswith(".dist-info")]
    print(f"  Vendored {len(pkg_dirs)} package directories")


def download_nltk_data(dest_dir: Path):
    """Download NLTK data (punkt_tab, wordnet) into a bundleable directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading NLTK data to {dest_dir}...")

    try:
        import nltk
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nltk"])
        import nltk

    for package in ["punkt_tab", "wordnet"]:
        nltk.download(package, download_dir=str(dest_dir), quiet=True)
        print(f"    Downloaded: {package}")

    print(f"  NLTK data ready.")


def build_mcp_zip(output_name: str = "sumbench_mcp.zip", cleanup: bool = True,
                  include_wheels: bool = False, include_deps: bool = False,
                  platform: str = None,
                  python_version: str = None, torch_variant: str = "cpu",
                  include_spacy_model: bool = False, include_nltk_data: bool = False):
    """
    Build a zip file containing the MCP server and all dependencies.

    Args:
        output_name: Name of the output zip file.
        cleanup: Whether to remove the temp directory after zipping.
        include_wheels: Download and bundle wheel files for offline install.
        include_deps: Install packages into vendor/ directory (no .whl files).
        platform: Target platform for wheels (e.g., manylinux2014_x86_64).
        python_version: Target Python version (e.g., 3.11).
        torch_variant: PyTorch variant (cpu, cu118, cu121, cu124).
        include_spacy_model: Bundle en_core_web_sm spaCy model.
        include_nltk_data: Bundle NLTK data (punkt_tab, wordnet).
    """
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    dist_dir = base_dir / "dist_mcp"

    print(f"Building MCP bundle...")
    print(f"  Project root: {project_root}")
    print(f"  Output: {output_name}")
    if include_wheels:
        print(f"  Mode: AIRGAPPED (bundling wheels)")

    # Clean old builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    # Copy server.py
    server_src = base_dir / "server.py"
    if not server_src.exists():
        raise FileNotFoundError(f"server.py not found at {server_src}")
    shutil.copy(server_src, dist_dir / "server.py")
    print(f"  Copied: server.py")

    # Copy envs.json
    envs_src = base_dir / "envs.json"
    if envs_src.exists():
        shutil.copy(envs_src, dist_dir / "envs.json")
        print(f"  Copied: envs.json")
    else:
        print(f"  Warning: envs.json not found at {envs_src}")

    # Copy requirements file from project root
    # Prefer requirements-mcp.txt (lightweight, no UI/torch) for MCP bundles;
    # fall back to requirements.txt if it doesn't exist.
    req_src = project_root / "requirements-mcp.txt"
    if not req_src.exists():
        req_src = project_root / "requirements.txt"
        print(f"  Warning: requirements-mcp.txt not found, using requirements.txt")
    if req_src.exists():
        shutil.copy(req_src, dist_dir / "requirements.txt")
        print(f"  Copied: {req_src.name} -> requirements.txt")
    else:
        print(f"  Warning: requirements not found at {req_src}")

    # Copy evaluators directory (flattened - not nested in src/)
    # This allows the bundled server to import via `from evaluators.tool_logic import ...`
    evaluators_dir = project_root / "src" / "evaluators"
    if not evaluators_dir.exists():
        raise FileNotFoundError(f"evaluators directory not found at {evaluators_dir}")

    def copy_filter(directory, files):
        """Filter out excluded files and directories."""
        return [f for f in files if should_exclude(f)]

    shutil.copytree(evaluators_dir, dist_dir / "evaluators", ignore=copy_filter)
    print(f"  Copied: evaluators/ directory")

    # Bundle dependencies for airgapped environments
    if include_deps:
        # Install packages directly into vendor/ (no .whl files)
        req_path = dist_dir / "requirements.txt"
        if req_path.exists():
            vendor_dir = dist_dir / "vendor"
            install_vendored_deps(req_path, vendor_dir)
        else:
            print(f"  Warning: Cannot install deps - requirements.txt not found")
    elif include_wheels:
        # Download wheel files into wheels/
        req_path = dist_dir / "requirements.txt"
        if req_path.exists():
            wheels_dir = dist_dir / "wheels"
            download_wheels(req_path, wheels_dir, platform, python_version, torch_variant)

            if include_spacy_model:
                download_spacy_model(wheels_dir)
        else:
            print(f"  Warning: Cannot download wheels - requirements.txt not found")

    # Download and bundle NLTK data
    if include_nltk_data:
        nltk_dir = dist_dir / "nltk_data"
        download_nltk_data(nltk_dir)

    # Create zip file
    zip_path = base_dir / output_name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dist_dir):
            # Filter directories in-place to skip excluded ones
            dirs[:] = [d for d in dirs if not should_exclude(d)]

            for file in files:
                if should_exclude(file):
                    continue
                file_path = Path(root) / file
                arc_name = file_path.relative_to(dist_dir)
                zipf.write(file_path, arc_name)

    # Get zip size
    zip_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Created: {output_name} ({zip_size:.2f} MB)")

    # Cleanup temp directory
    if cleanup:
        shutil.rmtree(dist_dir)
        print(f"  Cleaned up: {dist_dir}")

    print(f"Done! Bundle ready at: {zip_path}")
    return zip_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bundle MCP server for deployment")
    parser.add_argument("--output", "-o", default="sumbench_mcp.zip",
                        help="Output zip filename")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep the temp directory after bundling")

    # Airgapped environment options
    airgap = parser.add_argument_group("airgapped environment options")
    airgap.add_argument("--include-deps", action="store_true",
                        help="Install packages into vendor/ directory (no .whl files, platform-compatible)")
    airgap.add_argument("--include-wheels", action="store_true",
                        help="Download and bundle wheel files for offline install")
    airgap.add_argument("--platform", default=None,
                        help="Target platform (e.g., manylinux2014_x86_64, macosx_11_0_arm64)")
    airgap.add_argument("--python-version", default=None,
                        help="Target Python version (e.g., 3.11)")
    airgap.add_argument("--torch-variant", default="cpu",
                        choices=["cpu", "cu118", "cu121", "cu124"],
                        help="PyTorch variant to bundle (default: cpu)")
    airgap.add_argument("--include-spacy-model", action="store_true",
                        help="Bundle en_core_web_sm spaCy model")
    airgap.add_argument("--include-nltk-data", action="store_true",
                        help="Bundle NLTK data (punkt_tab, wordnet)")

    args = parser.parse_args()
    build_mcp_zip(
        output_name=args.output,
        cleanup=not args.no_cleanup,
        include_wheels=args.include_wheels,
        include_deps=args.include_deps,
        platform=args.platform,
        python_version=args.python_version,
        torch_variant=args.torch_variant,
        include_spacy_model=args.include_spacy_model,
        include_nltk_data=args.include_nltk_data,
    )
