#!/usr/bin/env python3
"""Project setup script for q-reply.

This script automates the local deployment of the q-reply project.
It downloads the project archive, extracts files, sets up a virtual environment,
and installs dependencies.

Usage:
    python setup.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import NoReturn
from urllib.error import URLError
from urllib.request import urlretrieve
import zipfile


# Project configuration
PROJECT_URL = "https://github.com/yesgenius/q-reply/archive/refs/heads/main.zip"
VENV_DIR = ".venv"
TEMP_ARCHIVE = "project_archive.zip"


def print_status(message: str, level: str = "INFO") -> None:
    """Print formatted status message.

    Args:
        message: Status message to display.
        level: Message level (INFO, SUCCESS, ERROR, WARNING).
    """
    symbols = {"INFO": "ℹ️ ", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️ "}
    print(f"{symbols.get(level, '')} {message}")


def fail_fast(message: str) -> NoReturn:
    """Exit with error message following Fail Fast principle.

    Args:
        message: Error description.
    """
    print_status(message, "ERROR")
    sys.exit(1)


def download_archive(url: str, destination: str) -> None:
    """Download project archive from GitHub.

    Args:
        url: GitHub archive URL.
        destination: Local file path for downloaded archive.

    Raises:
        SystemExit: If download fails.
    """
    print_status(f"Downloading project from: {url}")

    try:
        urlretrieve(url, destination)
        print_status("Download completed", "SUCCESS")
    except URLError as e:
        fail_fast(f"Failed to download archive: {e}")
    except Exception as e:
        fail_fast(f"Unexpected error during download: {e}")


def merge_directories(src: Path, dst: Path) -> None:
    """Recursively merge source directory into destination.

    Files are overwritten, directories are merged without deletion.

    Args:
        src: Source directory path.
        dst: Destination directory path.
    """
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        src_path = src / item.name
        dst_path = dst / item.name

        if src_path.is_dir():
            # Recursively merge subdirectories
            merge_directories(src_path, dst_path)
        else:
            # Overwrite file (remove old if exists, then copy new)
            if dst_path.exists():
                dst_path.unlink()
            shutil.copy2(src_path, dst_path)


def extract_archive(archive_path: str, target_dir: Path) -> None:
    """Extract and merge archive contents into current directory.

    Files are overwritten, directories are merged (not replaced).
    Existing files not in archive remain untouched.

    Args:
        archive_path: Path to zip archive.
        target_dir: Directory where to extract files.

    Raises:
        SystemExit: If extraction fails.
    """
    print_status("Extracting and merging archive contents...")

    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Get root folder name in archive
            archive_members = zip_ref.namelist()
            if not archive_members:
                fail_fast("Archive is empty")

            # Identify root folder in archive (e.g., 'q-reply-main')
            root_folder = archive_members[0].split("/")[0]

            # Extract to temporary directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)

                # Source directory in temp location
                source_dir = Path(temp_dir) / root_folder

                # Merge contents intelligently
                for item in source_dir.iterdir():
                    src_item = source_dir / item.name
                    dst_item = target_dir / item.name

                    if src_item.is_dir():
                        # Merge directory contents recursively
                        merge_directories(src_item, dst_item)
                    else:
                        # Overwrite single file
                        if dst_item.exists():
                            dst_item.unlink()
                        shutil.copy2(src_item, dst_item)

        print_status("Archive merged successfully", "SUCCESS")

    except zipfile.BadZipFile:
        fail_fast("Invalid or corrupted archive file")
    except Exception as e:
        fail_fast(f"Failed to extract archive: {e}")


def setup_virtual_environment(venv_path: str) -> None:
    """Create and activate virtual environment.

    Args:
        venv_path: Path for virtual environment directory.

    Raises:
        SystemExit: If venv creation fails.
    """
    print_status(f"Creating virtual environment: {venv_path}")

    # Remove existing venv if present
    if os.path.exists(venv_path):
        print_status("Removing existing virtual environment", "WARNING")
        shutil.rmtree(venv_path)

    try:
        # Create virtual environment
        subprocess.run(
            [sys.executable, "-m", "venv", venv_path], check=True, capture_output=True, text=True
        )
        print_status("Virtual environment created", "SUCCESS")

    except subprocess.CalledProcessError as e:
        fail_fast(f"Failed to create virtual environment: {e.stderr}")
    except Exception as e:
        fail_fast(f"Unexpected error creating venv: {e}")


def install_requirements(venv_path: str, requirements_file: str) -> None:
    """Install requirements using pip in virtual environment.

    Args:
        venv_path: Path to virtual environment.
        requirements_file: Path to requirements.txt file.

    Raises:
        SystemExit: If installation fails.
    """
    print_status("Installing dependencies from requirements.txt")

    # Determine python executable path based on OS
    if sys.platform == "win32":
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_path, "bin", "python")

    try:
        # Upgrade pip using python -m pip (more reliable on Windows)
        print_status("Upgrading pip...")
        result = subprocess.run(
            [python_executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=False,
            capture_output=True,
            text=True,
        )

        # Don't fail if pip upgrade has warnings
        if result.returncode != 0 and "ERROR:" in result.stderr:
            fail_fast(f"Failed to upgrade pip: {result.stderr}")

        # Install requirements
        print_status("Installing packages from requirements.txt...")
        subprocess.run(
            [python_executable, "-m", "pip", "install", "-r", requirements_file],
            check=True,
            capture_output=True,
            text=True,
        )
        print_status("Dependencies installed", "SUCCESS")

    except subprocess.CalledProcessError as e:
        fail_fast(f"Failed to install dependencies: {e.stderr}")
    except Exception as e:
        fail_fast(f"Unexpected error during installation: {e}")


def cleanup_archive(archive_path: str) -> None:
    """Remove temporary archive file.

    Args:
        archive_path: Path to archive file to remove.
    """
    try:
        if os.path.exists(archive_path):
            os.remove(archive_path)
            print_status("Cleaned up temporary files", "SUCCESS")
    except Exception as e:
        print_status(f"Warning: Could not remove archive: {e}", "WARNING")


def get_activation_commands() -> list[str]:
    """Get platform-specific venv activation commands.

    Returns:
        List of activation command strings for current platform.
    """
    if sys.platform == "win32":
        return [
            f"{VENV_DIR}\\Scripts\\activate.bat  # Windows CMD",
            f"{VENV_DIR}\\Scripts\\Activate.ps1  # Windows PowerShell",
        ]
    return [f"source {VENV_DIR}/bin/activate"]


def main() -> None:
    """Main setup workflow with safe merge strategy."""
    print_status("Starting project setup", "INFO")
    print_status(f"Python version: {sys.version}", "INFO")
    print_status(f"Working directory: {os.getcwd()}", "INFO")

    current_dir = Path.cwd()

    try:
        # Step 1: Download archive
        download_archive(PROJECT_URL, TEMP_ARCHIVE)

        # Step 2: Extract and merge archive (preserves existing files not in archive)
        extract_archive(TEMP_ARCHIVE, current_dir)

        # Step 3: Check for requirements.txt and setup virtual environment
        requirements_path = current_dir / "requirements.txt"

        if requirements_path.exists():
            print_status("Found requirements.txt", "INFO")

            # Step 4: Setup fresh virtual environment
            setup_virtual_environment(VENV_DIR)

            # Step 5: Install dependencies
            install_requirements(VENV_DIR, str(requirements_path))

            print_status("Virtual environment configured with all dependencies", "SUCCESS")
        else:
            print_status("No requirements.txt found, skipping venv setup", "WARNING")

        # Step 6: Cleanup temporary files only
        cleanup_archive(TEMP_ARCHIVE)

        # Final status
        print("\n" + "=" * 50)
        print_status("Setup completed successfully!", "SUCCESS")

        if requirements_path.exists():
            print("\nTo activate the virtual environment, run:")

            # Display activation commands
            commands = get_activation_commands()
            for cmd in commands:
                print(f"  {cmd}")

        print("\nProject files have been merged safely.")
        print("Existing files not in the archive were preserved.")
        print("=" * 50)

    except KeyboardInterrupt:
        print_status("\nSetup interrupted by user", "WARNING")
        cleanup_archive(TEMP_ARCHIVE)
        sys.exit(1)
    except Exception as e:
        fail_fast(f"Setup failed with unexpected error: {e}")


if __name__ == "__main__":
    main()
