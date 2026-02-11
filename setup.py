"""
H2O.ai SumBench - One-shot setup script.

Creates a virtual environment, installs all Python dependencies,
downloads the spaCy language model, and fetches required NLTK data.
Works on macOS, Linux, and Windows.

Usage:
    python setup.py          # or python3 setup.py
"""

import os
import platform
import shutil
import subprocess
import sys
import venv

MIN_PYTHON = (3, 9)
VENV_DIR = ".venv"


def check_python_version():
    """Exit early if the Python version is too old."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
            f"(you have {platform.python_version()})."
        )


def venv_python():
    """Return the path to the Python executable inside the venv."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def pip_works(py):
    """Return True if `py -m pip --version` exits 0."""
    return subprocess.run(
        [py, "-m", "pip", "--version"],
        capture_output=True,
    ).returncode == 0


def create_venv():
    """Create a virtual environment if it doesn't already exist."""
    py = venv_python()

    # If venv exists, check it actually works
    if os.path.isfile(py):
        if pip_works(py):
            print(f"\n  Virtual environment already exists at {VENV_DIR}/")
            return
        # Broken venv — nuke and recreate
        print(f"\n  Existing {VENV_DIR}/ is broken (pip not functional). Recreating ...")
        shutil.rmtree(VENV_DIR)

    print(f"\n  Creating virtual environment in {VENV_DIR}/ ...")
    venv.create(VENV_DIR, with_pip=True)

    if not os.path.isfile(py):
        sys.exit(f"ERROR: Failed to create virtual environment (expected {py}).")

    # Verify pip; bootstrap if needed
    if not pip_works(py):
        print("  pip not available — bootstrapping with ensurepip ...")
        result = subprocess.run([py, "-m", "ensurepip", "--upgrade"], capture_output=True)
        if result.returncode != 0 or not pip_works(py):
            sys.exit(
                "ERROR: Could not install pip in the virtual environment.\n"
                "  macOS: try `brew install python3` and re-run with that Python.\n"
                "  Linux: try `sudo apt install python3-venv` (Debian/Ubuntu)."
            )

    print("  Created.")


def run(description, cmd):
    """Run a command, printing status and exiting on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed (exit code {result.returncode}).")
        sys.exit(result.returncode)
    print(f"\n  Done.\n")


def verify_install(py):
    """Quick smoke-test: can the venv Python import the key packages?"""
    check = subprocess.run(
        [py, "-c", "import streamlit, torch, transformers"],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        print("\nERROR: Post-install verification failed.")
        print("  The following packages could not be imported:")
        print(f"  {check.stderr.strip()}")
        print("\n  Try deleting .venv/ and re-running: python setup.py")
        sys.exit(1)


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print("\nH2O.ai SumBench Setup")
    print(f"  Python  : {platform.python_version()}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    # 0. Version gate
    check_python_version()

    # 1. Create venv
    print(f"\n{'=' * 60}")
    print(f"  Creating virtual environment")
    print(f"{'=' * 60}")
    create_venv()
    py = venv_python()

    # 2. Upgrade pip inside venv
    run(
        "Upgrading pip",
        [py, "-m", "pip", "install", "--upgrade", "pip"],
    )

    # 3. Install dependencies
    run(
        "Installing Python dependencies (requirements.txt)",
        [py, "-m", "pip", "install", "-r", "requirements.txt"],
    )

    # 4. spaCy model
    run(
        "Downloading spaCy model (en_core_web_sm)",
        [py, "-m", "spacy", "download", "en_core_web_sm"],
    )

    # 5. NLTK data
    run(
        "Downloading NLTK data (punkt_tab)",
        [py, "-c", "import nltk; nltk.download('punkt_tab')"],
    )

    # 6. Verify key packages are importable
    verify_install(py)

    # Done
    if platform.system() == "Windows":
        activate = f"{VENV_DIR}\\Scripts\\activate"
    else:
        activate = f"source {VENV_DIR}/bin/activate"

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Activate the virtual environment:")
    print(f"       {activate}")
    print()
    print("  2. Go to .env.example, add your H2OGPTE API key, and rename the file to .env")
    print()
    print("  3. Launch the app:")
    print("       streamlit run ui/app.py")
    print()


if __name__ == "__main__":
    main()
