import torch
from math import log2

# Dataset location
DATASET = "celeb_dataset"

# Checkpoints
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"

# Use CPU
DEVICE = "cpu"

# Training options
SAVE_MODEL = True
LOAD_MODEL = False

# Learning rate
LEARNING_RATE = 1e-3

# Progressive training start size
START_TRAIN_AT_IMG_SIZE = 4

# Batch sizes for each resolution (CPU friendly)
BATCH_SIZES = [16, 16, 16, 8, 8, 4, 4, 2, 1]

# Image channels (RGB)
CHANNELS_IMG = 3

# Latent vector size
Z_DIM = 256

# Feature map size
IN_CHANNELS = 256

# Critic updates
CRITIC_ITERATIONS = 1

# Gradient penalty coefficient
LAMBDA_GP = 10

# Epochs per resolution (reduced for CPU)
PROGRESSIVE_EPOCHS = [5] * len(BATCH_SIZES)

# Fixed noise for TensorBoard visualization
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)

# CPU workers
NUM_WORKERS = 0