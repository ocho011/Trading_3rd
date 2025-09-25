#!/usr/bin/env python3
"""
Comprehensive lint fix script for trading bot project.
This script systematically fixes common lint errors.
"""

import os
import re
import subprocess
from pathlib import Path


def fix_missing_newlines(project_root: str) -> None:
    """Add missing newlines at end of Python files."""
    python_files = []
    for root, _, files in os.walk(project_root):
        if 'site-packages' in root or '.git' in root or '__pycache__' in root:
            continue
        python_files.extend([
            os.path.join(root, f) for f in files if f.endswith('.py')
        ])

    for file_path in python_files:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            if content and not content.endswith(b'\n'):
                with open(file_path, 'ab') as f:
                    f.write(b'\n')
                print(f"Fixed missing newline: {file_path}")
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")


def remove_unused_imports() -> None:
    """Remove specific unused imports we know about."""
    fixes = {
        '/Users/osangwon/github/thirdtry/trading_bot/notification/enhanced_discord_client.py': [
            'from urllib.parse import parse_qs, urlparse',
        ],
        '/Users/osangwon/github/thirdtry/trading_bot/portfolio_manager/portfolio_manager.py': [
            'from typing import List, Union',
            'from trading_bot.portfolio_manager.position import PositionStatus',
        ]
    }

    for file_path, imports_to_remove in fixes.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                original_content = content
                for import_line in imports_to_remove:
                    content = content.replace(import_line + '\n', '')

                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"Removed unused imports from: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def run_black_formatting(project_root: str) -> None:
    """Run black formatter on the project."""
    try:
        result = subprocess.run([
            'python3', '-m', 'black',
            '--line-length', '88',
            'trading_bot/'
        ], cwd=project_root, capture_output=True, text=True)

        if result.returncode == 0:
            print("Black formatting completed successfully")
            print(result.stdout)
        else:
            print("Black formatting failed:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running black: {e}")


def run_isort(project_root: str) -> None:
    """Run isort on the project."""
    try:
        result = subprocess.run([
            'python3', '-m', 'isort',
            'trading_bot/',
            '--profile', 'black'
        ], cwd=project_root, capture_output=True, text=True)

        if result.returncode == 0:
            print("Import sorting completed successfully")
            print(result.stdout)
        else:
            print("Import sorting failed:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running isort: {e}")


if __name__ == "__main__":
    project_root = "/Users/osangwon/github/thirdtry"

    print("Starting comprehensive lint fixes...")

    # Fix missing newlines
    print("\n1. Fixing missing newlines...")
    fix_missing_newlines(project_root)

    # Remove known unused imports
    print("\n2. Removing unused imports...")
    remove_unused_imports()

    # Run black formatting
    print("\n3. Running black formatter...")
    run_black_formatting(project_root)

    # Run isort
    print("\n4. Sorting imports...")
    run_isort(project_root)

    print("\nLint fixes completed!")
