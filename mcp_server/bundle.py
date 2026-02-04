"""
Bundle MCP server and dependencies into a deployable zip file.
"""

import os
import shutil
import zipfile
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


def should_exclude(path: str) -> bool:
    """Check if a path should be excluded from the bundle."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path:
            return True
    return False


def build_mcp_zip(output_name: str = "sum_omni_eval_mcp.zip", cleanup: bool = True):
    """
    Build a zip file containing the MCP server and all dependencies.

    Args:
        output_name: Name of the output zip file.
        cleanup: Whether to remove the temp directory after zipping.
    """
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    dist_dir = base_dir / "dist_mcp"

    print(f"Building MCP bundle...")
    print(f"  Project root: {project_root}")
    print(f"  Output: {output_name}")

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

    # Copy requirements.txt from project root
    req_src = project_root / "requirements.txt"
    if req_src.exists():
        shutil.copy(req_src, dist_dir / "requirements.txt")
        print(f"  Copied: requirements.txt")
    else:
        print(f"  Warning: requirements.txt not found at {req_src}")

    # Copy src directory (excluding __pycache__)
    src_dir = project_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"src directory not found at {src_dir}")

    def copy_filter(directory, files):
        """Filter out excluded files and directories."""
        return [f for f in files if should_exclude(f)]

    shutil.copytree(src_dir, dist_dir / "src", ignore=copy_filter)
    print(f"  Copied: src/ directory")

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
    parser.add_argument("--output", "-o", default="sum_omni_eval_mcp.zip",
                        help="Output zip filename")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep the temp directory after bundling")

    args = parser.parse_args()
    build_mcp_zip(output_name=args.output, cleanup=not args.no_cleanup)
