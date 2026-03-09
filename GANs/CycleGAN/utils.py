import os
import random
import torch
import numpy as np
import config


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """
    Save model and optimizer state
    """
    print("=> Saving checkpoint")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load model and optimizer state
    """
    print("=> Loading checkpoint")

    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Reset learning rate (important)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    """
    Make experiments reproducible
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU deterministic behavior
    torch.use_deterministic_algorithms(True)