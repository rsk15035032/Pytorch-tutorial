# CycleGAN Implementation (Horse ↔ Zebra) – PyTorch

This repository contains a **PyTorch implementation of CycleGAN** for **unpaired image-to-image translation** between **horse and zebra images**. The implementation follows the architecture proposed in the CycleGAN paper and is inspired by tutorials from **:contentReference[oaicite:0]{index=0}**. The project is designed to work **both on CPU and GPU**, and it demonstrates how to train a CycleGAN model using an **unpaired dataset**.

The goal of CycleGAN is to learn image translation **without paired examples**. Instead of requiring matching images from two domains, the model learns mappings using **cycle consistency**.

Example translations:

- Horse → Zebra  
- Zebra → Horse  

---

# Overview

CycleGAN consists of **two generators and two discriminators**.

### Generators

- **G_Z** : Horse → Zebra  
- **G_H** : Zebra → Horse  

### Discriminators

- **D_Z** : Determines if a zebra image is real or generated  
- **D_H** : Determines if a horse image is real or generated  

---

# Loss Functions

CycleGAN training is based on three losses.

## 1. Adversarial Loss

Encourages generators to produce realistic images.

Example:
        Generator tries to fool the discriminator
        Discriminator tries to detect fake images


---

## 2. Cycle Consistency Loss

Ensures the translated image can be converted back to the original.

Example:
        Horse → Zebra → Horse ≈ Original Horse


---

## 3. Identity Loss

Encourages generators to preserve **color and structure**.

Example:
        Horse → Horse should remain Horse
        Zebra → Zebra should remain Zebra


---

# Project Structure

The project is organized as follows:

CycleGAN/
│
├── train.py
├── dataset.py
├── config.py
├── utils.py
├── generator.py
├── discriminator.py
│
├── data/
│ ├── train/
│ │ ├── horses/
│ │ └── zebras/
│ │
│ └── val/
│ ├── horses/
│ └── zebras/
│
├── saved_images/
│
├── gen_horse.pth.tar
├── gen_zebra.pth.tar
├── disc_horse.pth.tar
└── disc_zebra.pth.tar


Explanation of main files:

| File | Purpose |
|-----|------|
| train.py | Main training script |
| dataset.py | Custom dataset loader |
| generator.py | Generator architecture |
| discriminator.py | Discriminator architecture |
| utils.py | Utility functions (checkpoint saving/loading) |
| config.py | Training configuration |

---

# Dataset Structure

The dataset must follow this structure:

data/
│
├── train/
│ ├── horses/
│ │ horse1.jpg
│ │ horse2.jpg
│ │ horse3.jpg
│ │ ...
│ │
│ └── zebras/
│ zebra1.jpg
│ zebra2.jpg
│ zebra3.jpg
│ ...
│
└── val/
├── horses/
└── zebras/


Important notes:

- Images **do not need to be paired**
- The dataset simply contains images from two domains
- CycleGAN learns the translation automatically

---

# Installation

Clone the repository:
                    git clone https://github.com/rsk15035032/pytorch-tutorial/GANs/CycleGAN.git

                    cd CycleGAN


Install dependencies:
                    pip install torch torchvision albumentations tqdm pillow numpy


---

# Training

Start training using:


python train.py


During training:

- Generated images are saved in `saved_images/`
- Model checkpoints are saved automatically
- Training progress is shown using `tqdm`

---

# Training Configuration

Default configuration used in this project:


Epochs: 20
Batch Size: 1
Learning Rate: 2e-4
Image Size: 256 × 256


The model in this repository was trained for **20 epochs** only for demonstration and testing.

For **better results**, it is recommended to train for:


100 – 200 epochs


GAN models generally require **longer training** to produce high-quality results.

---

# Image Preprocessing

Image augmentation is implemented using **Albumentations**.

Transformations applied:

- Resize to 256×256
- Random horizontal flip
- Normalization
- Conversion to PyTorch tensor

Normalization parameters:


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


This converts pixel values from:


[0,255] → [-1,1]


which matches the **tanh activation** used in the generator output.

---

# CPU Training

This implementation is optimized to work on **CPU machines**.

Recommended settings inside `config.py`:


DEVICE = "cpu"
NUM_WORKERS = 0
PIN_MEMORY = False


Notes:

- CPU training will be **slower than GPU**
- Suitable for testing or small experiments
- Works well on standard laptops

---

# GPU Training

If you have a GPU, you can significantly speed up training.

Modify the following parameters in `config.py`:


DEVICE = "cuda"
NUM_WORKERS = 4
PIN_MEMORY = True


You may also enable **mixed precision training** in `train.py`:


torch.cuda.amp.autocast()
torch.cuda.amp.GradScaler()


Benefits of GPU training:

- Faster training
- Larger batch sizes
- Reduced training time

---

# Model Checkpoints

The following checkpoints are saved during training:


gen_horse.pth.tar
gen_zebra.pth.tar
disc_horse.pth.tar
disc_zebra.pth.tar


Each checkpoint stores:

- Model weights
- Optimizer states

To resume training, set:


LOAD_MODEL = True


inside `config.py`.

---

# Generated Images

Generated images are saved periodically during training.

Directory:


saved_images/


Example outputs:


saved_images/
│
├── horse_0.png
├── zebra_0.png
├── horse_200.png
└── zebra_200.png


These images help visualize the model's progress during training.

---

# Key Features

- PyTorch implementation of CycleGAN
- Works with **unpaired datasets**
- Supports **CPU and GPU training**
- Automatic checkpoint saving
- Image generation during training
- Albumentations-based preprocessing

---

# Possible Improvements

Future improvements for this project may include:

- Image buffer for discriminator stability
- Learning rate decay scheduler
- TensorBoard visualization
- Training with larger datasets
- Training for 100–200 epochs
- Experimenting with different architectures

---

# Acknowledgment

This implementation is inspired by the deep learning tutorials of **:contentReference[oaicite:1]{index=1}**, whose educational content helped guide the structure of this project.

---

# License

This project is open-source and intended for **educational and research purposes**.
