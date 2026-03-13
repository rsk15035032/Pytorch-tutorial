# Configuration

This project uses a centralized config.py file to manage training parameters, dataset preprocessing, and model configuration. Centralizing these settings makes experiments reproducible and simplifies modification of hyperparameters during training.

## The configuration file defines:

- Device selection (CPU/GPU)

- Training hyperparameters

- Image resolution settings

- Data augmentation pipelines

- Model checkpoint paths

- Device Setup

The training device is automatically selected depending on whether a CUDA GPU is available.

### DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

If a GPU is available, training will utilize CUDA acceleration. Otherwise, the model runs on CPU.

Training Hyperparameters

These parameters control the training behavior of the generator and discriminator networks.

| Parameter      | Description                              | Default |
|---------------|------------------------------------------|--------|
| LEARNING_RATE | Learning rate used by the Adam optimizer | 1e-4   |
| NUM_EPOCHS    | Total number of training epochs          | 100    |
| BATCH_SIZE    | Number of images per batch               | 4      |
| NUM_WORKERS   | Data loading worker processes            | 0      |

Generative Adversarial Networks typically train more stably with smaller batches, especially when training on CPU.

Image Resolution Settings

The model performs 4× super-resolution.

| Variable | Value | Description                         |
|----------|------:|-------------------------------------|
| HIGH_RES | 96    | Target high-resolution image size   |
| LOW_RES  | 24    | Input low-resolution image size     |

The relationship is:

LOW_RES = HIGH_RES // 4

Therefore:

24 × 24  →  96 × 96

The generator learns to reconstruct high-resolution images from low-resolution inputs.

Image Normalization

Images are normalized differently depending on their role in training.

High Resolution Images
A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

This scales images to the range:

[-1 , 1]

This normalization is required because the generator outputs images using:

tanh()
Low Resolution Images

Low resolution images are created using bicubic downsampling.

A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC)

Bicubic interpolation simulates realistic low-resolution images similar to real-world degradation.

Data Augmentation

To improve model generalization, several augmentations are applied during training.

Random Crop
Horizontal Flip
Random 90° Rotation

Example pipeline:

```python
both_transforms = A.Compose([
    A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

These augmentations increase dataset diversity without collecting additional images.

Checkpoint Management

The project supports saving and loading model checkpoints.

Variable	Purpose
LOAD_MODEL	Load pretrained weights
SAVE_MODEL	Save checkpoints during training

### Checkpoint files:

gen.pth.tar
disc.pth.tar

These store:

Model weights

Optimizer state

Training progress

Test Transform

During evaluation, images are only normalized without augmentation.

```python
test_transform = A.Compose([
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2(),
])

This ensures deterministic outputs during testing.

Performance Tips (CPU Training)

If training on CPU, consider adjusting the following parameters:

| Setting     | Recommendation             |
|-------------|----------------------------|
| BATCH_SIZE  | Use `2–4`                  |
| NUM_WORKERS | Use `0`                    |
| HIGH_RES    | Reduce to `64` for faster training |

Reducing resolution can significantly improve training speed without drastically affecting results.

## Implementation Reference

This project follows the methodology introduced in the paper:

**Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**

The architecture combines:

- **Residual Learning**
- **Perceptual Loss**
- **Adversarial Training**

to generate **photo-realistic high-resolution images from low-resolution inputs**.