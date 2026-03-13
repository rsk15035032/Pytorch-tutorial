import torch
import config
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import MyImageFolder
from model import Generator, Discriminator
from loss import VGGLoss
from utils import load_checkpoint, save_checkpoint, plot_examples


if config.DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):

    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):

        low_res = low_res.to(config.DEVICE)
        high_res = high_res.to(config.DEVICE)

        # ---------------------
        # Train Discriminator
        # ---------------------

        fake = gen(low_res)

        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())

        real_labels = torch.ones_like(disc_real)
        fake_labels = torch.zeros_like(disc_fake)

        disc_loss_real = bce(disc_real, real_labels)
        disc_loss_fake = bce(disc_fake, fake_labels)

        loss_disc = disc_loss_real + disc_loss_fake

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ---------------------
        # Train Generator
        # ---------------------

        disc_fake = disc(fake)

        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))

        perceptual_loss = 0.006 * vgg_loss(fake, high_res)

        pixel_loss = mse(fake, high_res)

        gen_loss = pixel_loss + perceptual_loss + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        loop.set_postfix(
            d_loss=loss_disc.item(),
            g_loss=gen_loss.item(),
        )


def main():

    dataset = MyImageFolder(root_dir="new_data/")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)

    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")

        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        plot_examples("test_images/", gen)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()