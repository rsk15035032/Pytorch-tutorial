import torch
import config
from torch import nn, optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter


# ==========================================================
# GPU PERFORMANCE (DISABLED FOR CPU)
# ==========================================================

# Uncomment for GPU training
# torch.backends.cudnn.benchmark = True


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    writer,
    tb_step,
):

    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):

        low_res = low_res.to(config.DEVICE)
        high_res = high_res.to(config.DEVICE)

        # ---------------------
        # Train Discriminator
        # ---------------------

        fake = gen(low_res)

        critic_real = disc(high_res)
        critic_fake = disc(fake.detach())

        gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)

        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + config.LAMBDA_GP * gp
        )

        opt_disc.zero_grad()
        loss_critic.backward()
        opt_disc.step()

        # ---------------------
        # Train Generator
        # ---------------------

        l1_loss = 1e-2 * l1(fake, high_res)
        adversarial_loss = 5e-3 * -torch.mean(disc(fake))
        loss_for_vgg = vgg_loss(fake, high_res)

        gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # ---------------------
        # TensorBoard Logging
        # ---------------------

        writer.add_scalar("Critic Loss", loss_critic.item(), tb_step)
        writer.add_scalar("Generator Loss", gen_loss.item(), tb_step)

        tb_step += 1

        # Save example images
        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


# ==========================================================
# MAIN TRAINING LOOP
# ==========================================================

def main():

    dataset = MyImageFolder(root_dir="data/")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,  # CPU safe
    )

    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)

    initialize_weights(gen)

    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.0, 0.9),
    )

    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.0, 0.9),
    )

    writer = SummaryWriter("logs")
    tb_step = 0

    l1 = nn.L1Loss()
    vgg_loss = VGGLoss()

    gen.train()
    disc.train()

    # Load checkpoints
    if config.LOAD_MODEL:

        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )

        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    # Training loop
    for epoch in range(config.NUM_EPOCHS):

        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")

        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            writer,
            tb_step,
        )

        if config.SAVE_MODEL:

            save_checkpoint(
                gen,
                opt_gen,
                filename=config.CHECKPOINT_GEN,
            )

            save_checkpoint(
                disc,
                opt_disc,
                filename=config.CHECKPOINT_DISC,
            )


# ==========================================================
# INFERENCE MODE
# ==========================================================

if __name__ == "__main__":

    try_model = False

    if try_model:

        # Load trained generator and run inference

        gen = Generator(in_channels=3).to(config.DEVICE)

        opt_gen = optim.Adam(
            gen.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.0, 0.9),
        )

        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )

        plot_examples("test_images/", gen)

    else:

        main()