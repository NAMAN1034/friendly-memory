import os
import random
import numpy as np
import torch

# offsets so a seed in a crash log tells me which subsystem it was from
# decoding a hash. bump ga_base past 3000 if outer iters>20
ensemble_base = 100
panel_base = 200
ga_base = 1000
jump_base = 2000


def pin_seeds(run_seed, verbose=True):
    os.environ["PYTHONHASHSEED"] = str(run_seed)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    try:
        torch.mps.manual_seed(run_seed)
    except (AttributeError, RuntimeError):
        if verbose:
            print("no mps so embedding fallback to cpu")

    # esm2 pooling on mps is not deterministic, warn_only so a 6th replicate
    # doesn't stop over 12th decimal difference
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

def member_seed(run_seed, i):
    if not 0 <= i <10:
        raise ValueError(f"i={i} must be in [0,10)")
    return run_seed + ensemble_base + i


def start_panel_speed(run_seed):
    return run_seed + panel_base


def ga_seed(run_seed, outer_iter):
    return run_seed + ga_base + outer_iter


def jump_seed(run_seed, outer_iter):
    return run_seed + jump_base + outer_iter