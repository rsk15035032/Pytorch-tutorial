import random
import torch
import os
import numpy as np


def seed_everything(seed=42):
    """
    Set random seed everywhere to make experiments reproducible.
    
    This ensures that:
    - model initialization is the same every run
    - data shuffling is the same
    - numpy operations are the same
    - GPU results are reproducible
    """

    # ---------------------------
    # Python built-in randomness
    # ---------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)   # makes Python hashing deterministic
    random.seed(seed)                          # sets seed for Python random module

    # ---------------------------
    # NumPy randomness
    # ---------------------------
    np.random.seed(seed)                       # sets seed for NumPy

    # ---------------------------
    # PyTorch (CPU)
    # ---------------------------
    torch.manual_seed(seed)                    # sets seed for CPU operations

    # ---------------------------
    # PyTorch (GPU)
    # ---------------------------
    torch.cuda.manual_seed(seed)               # sets seed for current GPU
    torch.cuda.manual_seed_all(seed)           # sets seed for all GPUs (if multiple GPUs are used)

    # ---------------------------
    # Make CUDA deterministic
    # ---------------------------
    torch.backends.cudnn.deterministic = True  # ensures same results every run
    torch.backends.cudnn.benchmark = False     # disables auto-optimization (which can introduce randomness)


# Call this before training
seed_everything()