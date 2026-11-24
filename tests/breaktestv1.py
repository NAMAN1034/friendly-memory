
"""
Phase 1 smoke test.
Checks the project environment, MPS availability, and
reproducibility of the seed functions.
Developed with AI assistance from Deepseek and reviewed by Naman Pradhan.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import platform
import sys

import numpy as np
import torch
import pandas as pd
import sklearn
import pymoo
import esm

from src.utils.seeding import (
    pin_seeds,
    member_seed,
    start_panel_speed,
    ga_seed,
    jump_seed,
)


def check_environment() -> None:
    """Print the versions of the main project dependencies."""
    print("=== Environment ===")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"NumPy    : {np.__version__}")
    print(f"PyTorch  : {torch.__version__}")
    print(f"MPS      : {torch.backends.mps.is_available()}")
    print(f"pymoo    : {pymoo.__version__}")
    print(f"sklearn  : {sklearn.__version__}")
    print(f"pandas   : {pd.__version__}")
    print(f"fair-esm : {getattr(esm, '__version__', 'installed')}")


def check_global_seeding() -> None:
    """Check that resetting the seed produces identical random values."""
    pin_seeds(0)

    numpy_first = np.random.rand(3)
    torch_first = torch.rand(3)

    pin_seeds(0)

    numpy_second = np.random.rand(3)
    torch_second = torch.rand(3)

    numpy_ok = np.allclose(numpy_first, numpy_second)
    torch_ok = torch.allclose(torch_first, torch_second)

    print("\n=== Reproducibility ===")
    print(f"NumPy reproducibility : {numpy_ok}")
    print(f"PyTorch reproducibility: {torch_ok}")

    if not numpy_ok or not torch_ok:
        raise RuntimeError("Global seed reproducibility check failed.")


def check_subsystem_seeds() -> None:
    """Check that each subsystem receives its expected seed."""
    run_seed = 0

    seeds = {
        "Ensemble member 3": member_seed(run_seed, 3),
        "Panel": start_panel_speed(run_seed),
        "NSGA-II iteration 1": ga_seed(run_seed, 1),
        "Jump iteration 1": jump_seed(run_seed, 1),
    }

    expected = {
        "Ensemble member 3": 103,
        "Panel": 200,
        "NSGA-II iteration 1": 1001,
        "Jump iteration 1": 2001,
    }

    print("\n=== Subsystem Seeds ===")

    for name, seed in seeds.items():
        print(f"{name:22}: {seed}")

        if seed != expected[name]:
            raise RuntimeError(f"Unexpected seed for {name}.")

    if len(set(seeds.values())) != len(seeds):
        raise RuntimeError("Two subsystems received the same seed.")


def check_seed_reproducibility() -> None:
    """Check that subsystem seeds produce repeatable random values."""
    seed = member_seed(0, 3)

    pin_seeds(seed)
    first_draw = np.random.rand()

    pin_seeds(seed)
    second_draw = np.random.rand()

    print("\n=== Subsystem Reproducibility ===")
    print(f"Seed       : {seed}")
    print(f"First draw : {first_draw:.6f}")
    print(f"Second draw: {second_draw:.6f}")

    if first_draw != second_draw:
        raise RuntimeError("Subsystem reproducibility check failed.")


def main() -> None:
    check_environment()
    check_global_seeding()
    check_subsystem_seeds()
    check_seed_reproducibility()

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
