import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from model import Generator, Critic, initialize_weights
from utils import gradient_penalty

# -----------------------------
# Hyperparameters
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
IMAGE_SIZE = 32
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 20
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
NUM_CLASSES = 10
EMBED_SIZE = 100

# -----------------------------
# TensorBoard
# -----------------------------
writer = SummaryWriter("runs/cWGAN_GP")

# -----------------------------
# Dataset (MNIST → 32x32)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transform,
    download=True,
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Models
# -----------------------------
gen = Generator(Z_DIM, NUM_CLASSES, CHANNELS_IMG, FEATURES_GEN, EMBED_SIZE).to(device)
critic = Critic(CHANNELS_IMG, NUM_CLASSES, FEATURES_CRITIC, IMAGE_SIZE).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

step = 0
fixed_noise = torch.randn(10, Z_DIM, 1, 1).to(device)
fixed_labels = torch.arange(0, 10).to(device)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(tqdm(loader)):
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.shape[0]

        # ---------------------
        # Train Critic
        # ---------------------
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)

            critic_real = critic(real, labels)
            critic_fake = critic(fake.detach(), labels)

            gp = gradient_penalty(critic, real, fake, labels, device)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
            )

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # ---------------------
        # Train Generator
        # ---------------------
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise, labels)
        output = critic(fake, labels)

        loss_gen = -torch.mean(output)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # ---------------------
        # TensorBoard Logging
        # ---------------------
        writer.add_scalar("Loss/Critic", loss_critic.item(), step)
        writer.add_scalar("Loss/Generator", loss_gen.item(), step)
        writer.add_scalar("Loss/GP", gp.item(), step)

        if batch_idx % 200 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] "
                f"Loss Critic: {loss_critic:.4f}, "
                f"Loss Gen: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise, fixed_labels)
                img_grid = make_grid(fake, normalize=True)
                writer.add_image("Generated Digits", img_grid, global_step=epoch)

        step += 1

print("Training Complete!")
writer.close()