"""
Training of WGAN-GP

"""
import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import gradient_penalty
from model import Discriminator, Generator, initialize_weights

# =========================
# Hyperparameters
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 30
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# =========================
# Transforms
# =========================
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.5] * CHANNELS_IMG, [0.5] * CHANNELS_IMG),
])

# =========================
# Dataset
# =========================
dataset = datasets.ImageFolder(
    root="Celebrity Faces Dataset",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,   # important for stable GP
)

# =========================
# Models
# =========================
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

initialize_weights(gen)
initialize_weights(critic)

# =========================
# Optimizers
# =========================
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# =========================
# TensorBoard
# =========================
writer = SummaryWriter("logs/GAN_celeb")
writer_real = SummaryWriter("logs/GAN_celeb/real")
writer_fake = SummaryWriter("logs/GAN_celeb/fake")

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
step = 0

# =========================
# Training Loop
# =========================
gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(tqdm(loader)):

        real = real.to(device)
        cur_batch_size = real.size(0)

        # =========================
        # Train Critic
        # =========================
        for _ in range(CRITIC_ITERATIONS):

            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake.detach()).reshape(-1)

            gp = gradient_penalty(critic, real, fake.detach(), device)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
            )

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # =========================
        # Train Generator
        # =========================
        # regenerate fake for fresh graph
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # =========================
        # TensorBoard Logging
        # =========================
        writer.add_scalar("Loss/Critic", loss_critic.item(), step)
        writer.add_scalar("Loss/Generator", loss_gen.item(), step)

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] "
                f"Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_critic:.4f}, "
                f"Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake_samples = gen(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake_samples[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, step)
                writer_fake.add_image("Fake", img_grid_fake, step)

        step += 1


# =========================
# Close TensorBoard Writers
# =========================
writer.close()
writer_real.close()
writer_fake.close()