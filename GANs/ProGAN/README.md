# Progressive GAN (ProGAN) – PyTorch Implementation (CPU Compatible)

## Overview

This repository contains a **PyTorch implementation of Progressive Growing of GANs (ProGAN)** based on the paper:

> **"Progressive Growing of GANs for Improved Quality, Stability, and Variation"**  
> *Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (NVIDIA)*

The model progressively grows both the **Generator** and **Discriminator** during training. Instead of training directly on high-resolution images, the network starts with **very low resolution (4×4)** images and gradually increases the resolution up to **128×128 or higher**.

This implementation includes the major architectural ideas proposed in the original paper:

- Progressive growing of layers
- Wasserstein GAN loss with Gradient Penalty (WGAN-GP)
- Pixel Normalization
- Equalized Learning Rate
- Mini-Batch Standard Deviation layer
- Alpha blending (fade-in) between resolutions
- TensorBoard logging
- Checkpoint saving/loading
- CPU compatible training

The code is designed to be **clean, educational, and readable**, making it useful for **learning GAN theory and architecture**.

---

# Table of Contents

- Theory of GANs
- ProGAN Architecture
- Mathematical Formulation
- Key Techniques Implemented
- Project Structure
- Dataset Structure
- Installation
- Training
- TensorBoard Monitoring
- Generating Images
- Configuration
- References

---

# 1. Theory of Generative Adversarial Networks (GANs)

Generative Adversarial Networks consist of two neural networks:

### Generator (G)
The generator learns to map a **latent vector** \( z \) to an image.

\[
G(z) \rightarrow x_{fake}
\]

where

- \( z \sim \mathcal{N}(0,1) \)
- \( x_{fake} \) is a generated image

---

### Discriminator / Critic (D)

The discriminator tries to distinguish **real images from fake images**.

\[
D(x)
\]

Output:

- High score → real image
- Low score → fake image

---

### Adversarial Objective

GANs are trained as a **minimax game**:

\[
\min_G \max_D V(D,G)
\]

\[
V(D,G) =
\mathbb{E}_{x \sim p_{data}}[\log D(x)]
+
\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
\]

However, this loss often causes **unstable training**.

---

# 2. Wasserstein GAN (WGAN)

ProGAN uses **Wasserstein GAN** instead of the original GAN objective.

The critic tries to maximize:

\[
\mathbb{E}[D(x_{real})] - \mathbb{E}[D(x_{fake})]
\]

Generator tries to maximize:

\[
\mathbb{E}[D(G(z))]
\]

So the losses become:

### Critic Loss

\[
L_D =
-(\mathbb{E}[D(x_{real})] - \mathbb{E}[D(x_{fake})])
\]

### Generator Loss

\[
L_G =
-\mathbb{E}[D(G(z))]
\]

---

# 3. Gradient Penalty (WGAN-GP)

To enforce the **Lipschitz constraint**, WGAN-GP adds a gradient penalty term.

\[
L_{GP} =
\lambda \cdot
\mathbb{E}
\left[
(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2
\right]
\]

where

\[
\hat{x} = \epsilon x_{real} + (1-\epsilon)x_{fake}
\]

Final critic loss:

\[
L_{critic} =
-(\mathbb{E}[D(real)] - \mathbb{E}[D(fake)])
+
\lambda_{gp} L_{GP}
\]

---

# 4. Progressive Growing

Instead of training directly on large images, ProGAN **progressively increases resolution**.

Training sequence:


4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128


When transitioning between resolutions, a **fade-in mechanism** is used.

---

### Alpha Blending

During transition between resolutions:

\[
output =
\alpha \cdot new\_layer
+
(1-\alpha) \cdot old\_layer
\]

where

\[
0 \le \alpha \le 1
\]

This ensures **smooth transition between network architectures**.

---

# 5. Equalized Learning Rate

Instead of normal weight initialization, ProGAN scales weights during runtime.

Weight scaling factor:

\[
scale =
\sqrt{\frac{2}{fan\_in}}
\]

Forward pass:

\[
y = Conv(x \cdot scale)
\]

This helps stabilize GAN training.

---

# 6. Pixel Normalization

PixelNorm normalizes feature vectors across channels.

\[
PixelNorm(x) =
\frac{x}
{\sqrt{\frac{1}{N}\sum x_i^2 + \epsilon}}
\]

This stabilizes generator training.

---

# 7. Mini-Batch Standard Deviation

The discriminator receives additional information about batch variation.

Standard deviation across batch:

\[
\sigma =
std(x)
\]

This value is concatenated as an additional channel to the feature map.

Purpose:

- Prevent **mode collapse**
- Encourage diversity

---

# 8. Project Structure


progan
│
├── train.py
├── models.py
├── utils.py
├── config.py
│
├── celeb_dataset/
│ └── faces/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
│
├── logs/
├── saved_examples/
├── generator.pth
└── critic.pth


---

# 9. Dataset Format

Dataset must follow the **PyTorch ImageFolder structure**.

Example:


celeb_dataset/
faces/
image1.jpg
image2.jpg
image3.jpg


Only **one folder of images** is required.

---

# 10. Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/progan-pytorch
cd progan-pytorch

Install dependencies:

pip install torch torchvision tensorboard tqdm numpy scipy opencv-python
11. Training the Model

Start training:

python train.py

During training:

Generated images are saved in:

saved_examples/

Logs are stored in:

logs/

Model checkpoints are saved as:

generator.pth
critic.pth
12. Monitoring Training with TensorBoard

Run TensorBoard:

tensorboard --logdir=logs

Then open:

http://localhost:6006

You will see:

Generator loss

Critic loss

Real images

Generated images

13. Generating Images

To generate images after training:

generate_examples(generator, steps)

Generated images will be saved in:

saved_examples/
14. Configuration

Main parameters are defined in:

config.py

Example configuration:

START_TRAIN_AT_IMG_SIZE = 4
Z_DIM = 256
IN_CHANNELS = 256
LEARNING_RATE = 1e-3
BATCH_SIZES = [16,16,16,8,8,4]
PROGRESSIVE_EPOCHS = [5,5,5,5,5,5]
LAMBDA_GP = 10
DEVICE = "cpu"
15. Training Flow

The full training pipeline:

Initialize Generator
Initialize Discriminator

for resolution in [4,8,16,32,64,128]:

    load dataset resized to resolution

    for epoch:

        Train Critic
        Train Generator

        Update alpha fade-in
        Save checkpoints
        Log results
16. Notes on CPU Training

Training ProGAN on CPU is computationally expensive.

Recommended resolutions:

4×4
8×8
16×16
32×32
64×64

Training beyond 128×128 on CPU may take very long.

17. References

Paper:

Progressive Growing of GANs for Improved Quality, Stability, and Variation
Tero Karras et al. (2017)

https://arxiv.org/abs/1710.10196

Original NVIDIA implementation:

https://github.com/tkarras/progressive_growing_of_gans

18. Author

Implementation created for educational purposes and deep learning experimentation using PyTorch.

This project demonstrates how modern GAN techniques can be implemented in a clear and modular way.

19. License

This project is released under the MIT License.

20. Acknowledgment

Special thanks to the deep learning community and the original authors of the ProGAN paper for advancing generative modeling research.


---

If you want, I can also help you create a **much stronger GitHub version of this README** with:

- **Architecture diagrams**
- **training result images**
- **GAN loss graphs**
- **generator/discriminator architecture tables**

(which makes the project **10× more impressive for AI engineer portfolios**).