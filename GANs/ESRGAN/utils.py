import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image


# ==========================================================
# GRADIENT PENALTY (WGAN-GP)
# ==========================================================

def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape

    # Random weight term for interpolation between real and fake images
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Critic score for interpolated images
    mixed_scores = critic(interpolated_images)

    # Compute gradient of scores with respect to images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradient
    gradient = gradient.view(gradient.shape[0], -1)

    # L2 norm
    gradient_norm = gradient.norm(2, dim=1)

    # WGAN-GP penalty
    gp = torch.mean((gradient_norm - 1) ** 2)

    return gp


# ==========================================================
# SAVE MODEL CHECKPOINT
# ==========================================================

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)


# ==========================================================
# LOAD MODEL CHECKPOINT
# ==========================================================

def load_checkpoint(checkpoint_file, model, optimizer, lr):

    print("=> Loading checkpoint")

    # CPU safe loading
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Reset learning rate to current config
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ==========================================================
# GENERATE EXAMPLE IMAGES
# ==========================================================

def plot_examples(low_res_folder, gen):

    files = os.listdir(low_res_folder)

    gen.eval()

    for file in files:

        image = Image.open(os.path.join(low_res_folder, file))

        with torch.no_grad():

            img = config.test_transform(
                image=np.asarray(image)
            )["image"].unsqueeze(0).to(config.DEVICE)

            upscaled_img = gen(img)

        os.makedirs("saved", exist_ok=True)

        save_image(upscaled_img, f"saved/{file}")

    gen.train()