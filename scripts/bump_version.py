#!/usr/bin/env python3
"""Automatically bump the patch version in pyproject.toml."""

import re
import sys
from pathlib import Path


def bump_version(file_path: Path) -> str:
    """Read pyproject.toml, increment patch version, write back."""
    content = file_path.read_text()

    # Find version line
    version_pattern = r'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"'
    match = re.search(version_pattern, content)

    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    major, minor, patch = match.groups()
    new_patch = int(patch) + 1
    new_version = f"{major}.{minor}.{new_patch}"

    # Replace version
    new_content = re.sub(
        version_pattern,
        f'version = "{new_version}"',
        content
    )

    file_path.write_text(new_content)
    print(f"Version bumped: {major}.{minor}.{patch} -> {new_version}")
    return new_version


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"

    if not pyproject_file.exists():
        print(f"Error: {pyproject_file} not found")
        sys.exit(1)

    new_version = bump_version(pyproject_file)

    # Output version for capture
    print(f"NEW_VERSION={new_version}")
