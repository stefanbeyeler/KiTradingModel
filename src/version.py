"""Version configuration for KI Trading Model - derived from Git or environment."""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Base version
BASE_VERSION = "1.0.0"

# Cache for git info
_git_info_cache = None


def _load_build_info() -> dict | None:
    """Load build info from JSON file if available.

    This file is generated during Docker build by scripts/generate_build_info.sh
    """
    build_info_paths = [
        Path(__file__).parent / "build_info.json",
        Path("/app/src/build_info.json"),
    ]

    for path in build_info_paths:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

    return None


def _get_git_info() -> dict:
    """Get version info from git repository or environment variables.

    Priority:
    1. Build info JSON file (generated during Docker build)
    2. Environment variables (BUILD_VERSION, BUILD_COMMIT, BUILD_DATE, BUILD_NUMBER)
       - Set during Docker build or CI/CD
    3. Git repository (if available)
    4. Fallback values
    """
    global _git_info_cache

    if _git_info_cache is not None:
        return _git_info_cache

    # Check for build_info.json file first (Docker builds)
    build_info = _load_build_info()
    if build_info and build_info.get("commit"):
        commit_date = None
        if build_info.get("commit_date"):
            try:
                commit_date = datetime.fromisoformat(
                    build_info["commit_date"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        _git_info_cache = {
            "commit_hash": build_info.get("commit"),
            "commit_date": commit_date,
            "tag": build_info.get("tag") or None,
            "commit_count": build_info.get("commit_count", "0"),
        }
        return _git_info_cache

    # Check for environment variables (Docker/CI builds)
    env_commit = os.environ.get("BUILD_COMMIT")
    env_date = os.environ.get("BUILD_DATE")
    env_count = os.environ.get("BUILD_NUMBER")
    env_tag = os.environ.get("BUILD_TAG")

    if env_commit:
        # Use environment variables
        commit_date = None
        if env_date:
            try:
                commit_date = datetime.fromisoformat(env_date.replace("Z", "+00:00"))
            except ValueError:
                pass

        _git_info_cache = {
            "commit_hash": env_commit,
            "commit_date": commit_date,
            "tag": env_tag,
            "commit_count": env_count or "0"
        }
        return _git_info_cache

    # Fall back to Git if available
    try:
        # Get the repository root (where .git is located)
        repo_root = Path(__file__).parent.parent

        # Get short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else None

        # Get commit date/time (ISO format)
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_date_str = result.stdout.strip() if result.returncode == 0 else None

        # Parse commit date
        commit_date = None
        if commit_date_str:
            # Format: "2025-11-29 08:08:23 +0100"
            try:
                commit_date = datetime.strptime(commit_date_str[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        # Try to get tag if exists
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        tag = result.stdout.strip() if result.returncode == 0 else None

        # Get commit count for build number
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_count = result.stdout.strip() if result.returncode == 0 else "0"

        _git_info_cache = {
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "tag": tag,
            "commit_count": commit_count
        }

    except Exception:
        _git_info_cache = {
            "commit_hash": None,
            "commit_date": None,
            "tag": None,
            "commit_count": "0"
        }

    return _git_info_cache


def _build_version() -> str:
    """Build version string from git info or environment."""
    # Check for pre-built version from environment (highest priority)
    env_version = os.environ.get("BUILD_VERSION")
    if env_version:
        return env_version.lstrip("v")

    git_info = _get_git_info()

    # If there's a tag, use it
    if git_info["tag"]:
        return git_info["tag"].lstrip("v")

    # Build version string based on available info
    commit_hash = git_info["commit_hash"]
    commit_count = git_info["commit_count"] or "0"

    # If we have git info, include build number and commit hash
    if commit_hash:
        return f"{BASE_VERSION}.{commit_count}+{commit_hash}"

    # If commit count is available (but no hash), just use build number
    if commit_count and commit_count != "0":
        return f"{BASE_VERSION}.{commit_count}"

    # Fallback: just base version (no "unknown" suffix)
    return BASE_VERSION


def _get_release_date() -> str:
    """Get release date from git commit."""
    git_info = _get_git_info()

    if git_info["commit_date"]:
        return git_info["commit_date"].isoformat()

    return datetime.now().isoformat()


# Export version and release date
VERSION = _build_version()
RELEASE_DATE = _get_release_date()


def get_version_info() -> dict:
    """Get version information as dictionary."""
    git_info = _get_git_info()
    commit_date = git_info["commit_date"] or datetime.now()

    return {
        "version": f"v{VERSION}",
        "release_date": RELEASE_DATE,
        "release_date_formatted": commit_date.strftime("%d.%m.%Y %H:%M"),
        "commit_hash": git_info["commit_hash"],
        "build_number": git_info["commit_count"]
    }
