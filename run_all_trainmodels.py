#!/usr/bin/env python3
"""Run all Python programs inside the TrainModels folder."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    train_models_dir = root / "TrainModels"

    if not train_models_dir.is_dir():
        print(f"Error: {train_models_dir} is not a directory.")
        return 1

    python_files = sorted(
        p for p in train_models_dir.iterdir() if p.is_file() and p.suffix == ".py"
    )
    generate_plots = train_models_dir / "generate_plots.py"
    if generate_plots in python_files:
        python_files = [p for p in python_files if p != generate_plots] + [generate_plots]

    if not python_files:
        print("No Python files found in TrainModels.")
        return 0

    for script_path in python_files:
        print(f"\n=== Running {script_path.name} ===")
        result = subprocess.run(["python", str(script_path)], check=False)
        if result.returncode != 0:
            print(f"Error: {script_path.name} exited with code {result.returncode}.")
            return result.returncode

    print("\nAll TrainModels scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
