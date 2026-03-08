import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config

from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from utils import save_checkpoint, load_checkpoint, save_some_examples


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce):
    disc.train()
    gen.train()

    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # ============================
        # Train Discriminator
        # ============================

        y_fake = gen(x)

        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())

        loss_D_real = bce(D_real, torch.ones_like(D_real))
        loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))

        D_loss = (loss_D_real + loss_D_fake) / 2

        opt_disc.zero_grad()
        D_loss.backward()

        # Gradient clipping (important for CPU stability)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)

        opt_disc.step()

        # ============================
        # Train Generator
        # ============================

        D_fake = disc(x, y_fake)

        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA

        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)

        opt_gen.step()

        # ============================
        # Logging
        # ============================

        if idx % 10 == 0:
            loop.set_postfix(
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    print("Running on:", config.DEVICE)

    # ============================
    # Initialize models
    # ============================

    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)

    # ============================
    # Optimizers (CPU friendly)
    # ============================

    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # ============================
    # Load checkpoints (optional)
    # ============================

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # ============================
    # Dataset
    # ============================

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # IMPORTANT for Windows CPU
        pin_memory=False,       # No GPU → disable
    )

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ============================
    # Training Loop
    # ============================

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")

        train_fn(
            disc,
            gen,
            train_loader,
            opt_disc,
            opt_gen,
            L1_LOSS,
            BCE,
        )

        if config.SAVE_MODEL and (epoch + 1) % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()