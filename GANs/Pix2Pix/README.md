# Pix2Pix Image-to-Image Translation (Satellite → Map)

## Overview
This repository contains an implementation of **Pix2Pix**, a **Conditional Generative Adversarial Network (cGAN)** used for **image-to-image translation tasks**. In this project, Pix2Pix is trained on a **Satellite-to-Map dataset**, where the model learns to translate satellite images into map representations.

The dataset used in this project contains **paired images**, where each image consists of two halves:

- **Left Half → Input Image (Satellite Image)**
- **Right Half → Target Image (Map Image)**

During preprocessing, the image is split into these two halves so the model can learn the mapping between satellite imagery and map layouts.

This project focuses on a **CPU-compatible implementation** of Pix2Pix using **PyTorch**, making it possible to train the model even on systems without GPU support.

---

## Reference
This implementation is inspired by the excellent deep learning tutorials by **Aladdin Persson**. His tutorials on GANs and Pix2Pix provide a clear explanation of the architecture and training process.

You can find his tutorials here:

https://www.youtube.com/@AladdinPersson

The **GPU-optimized implementation** can be referred to from his work. This repository mainly focuses on a **CPU version of the training code**.

---

## Pix2Pix Architecture

Pix2Pix consists of two neural networks trained adversarially:

### Generator
The generator follows a **U-Net architecture**, which includes encoder-decoder layers with **skip connections**. These skip connections help preserve spatial information from the input image while generating the output.

The generator takes a **satellite image as input** and generates the **corresponding map image**.

### Discriminator
The discriminator is implemented as a **PatchGAN classifier**. Instead of classifying the entire image as real or fake, it classifies **small patches of the image**, which encourages the generator to produce sharper and more realistic outputs.

---

## Dataset Structure

Each dataset image contains both the input and target images concatenated horizontally.

Example structure:
| Satellite Image | Map Image |
|------Input------|---Target--|


Dataset directory example:

dataset/
│
├── train/
│ ├── image1.jpg
│ ├── image2.jpg
│ ├── image3.jpg
│
├── val/
│ ├── image1.jpg
│ ├── image2.jpg


During training, each image is split into:

- Left half → Input
- Right half → Target

---

## Project Structure

Pix2Pix/
│
├── dataset.py
├── model.py
├── train.py
├── utils.py
├── config.py
│
├── dataset/
│ ├── train/
│ └── val/
│
├── evaluation/
│
└── README.md



---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/pix2pix-satellite-map.git
cd pix2pix-satellite-map


Install required dependencies:

pip install torch torchvision albumentations tqdm tensorboard
Training the Model

Run the training script:

python train.py

During training:

Generated samples are saved in the evaluation folder.

Training metrics can be monitored using TensorBoard.

Run TensorBoard using:

tensorboard --logdir runs

Then open the following in your browser:

http://localhost:6006
Training Configuration

Typical training parameters used:

Batch Size: 16

Image Size: 256

Learning Rate: 2e-4

Generator: U-Net

Discriminator: PatchGAN

Loss Function: Adversarial Loss + L1 Loss

Training Epochs

For demonstration and faster experimentation on CPU, the model in this repository was trained for:

10 epochs

However, Pix2Pix models generally require longer training for better results.

For significantly improved output quality, it is recommended to train the model for:

200 epochs or more

depending on the dataset size.

Loss Function

Pix2Pix uses a combination of two loss functions.

Adversarial Loss

Encourages the generator to produce images that look realistic enough to fool the discriminator.

L1 Loss

Ensures that the generated image is close to the ground truth target image.

Total generator loss:

Loss_G = GAN_Loss + λ * L1_Loss

Where:

λ = 100
Example Results

After training, the model learns to translate satellite imagery into map-style representations.

Example translation:

Satellite Image → Generated Map

Generated outputs are automatically saved in the evaluation directory during training.

CPU vs GPU Training

This repository provides a CPU-compatible training implementation. Training GANs on CPU can be slow.

If you have access to a GPU, it is recommended to use a GPU version of the training code to significantly reduce training time.

Possible Improvements

Some ways to improve the model performance include:

Training for 200+ epochs

Using a larger dataset

Training with GPU acceleration

Increasing batch size

Applying data augmentation

Hyperparameter tuning

Applications of Pix2Pix

Pix2Pix can be applied to many image translation tasks such as:

Satellite → Map translation

Sketch → Photo generation

Black & White → Colorization

Day → Night conversion

Edge maps → Realistic images

Acknowledgement

Special thanks to Aladdin Persson for his excellent deep learning tutorials, which were very helpful in understanding and implementing the Pix2Pix architecture.

Author

Ravi Shankar Kumar

This project is created for educational and research purposes.