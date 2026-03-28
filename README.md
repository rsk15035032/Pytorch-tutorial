# PyTorch Deep Learning – Complete End-to-End Implementation

A clean and structured end-to-end PyTorch repository that covers everything from basic tensor operations to advanced deep learning architectures and real-world projects.

This repository is designed for beginners → intermediate → advanced AI/ML developers who want practical hands-on PyTorch implementations instead of only theory.

---

## Project Overview

This repository contains a step-by-step learning roadmap in PyTorch, starting from:

- Tensor fundamentals  
- Neural networks (CNN, RNN, LSTM)  
- Transfer learning & fine-tuning  
- Computer vision projects  
- NLP projects (Seq2Seq, Transformer, Translation)  
- Generative models (GAN family)  
- Object detection (YOLOv1 & YOLOv3)  
- Advanced architectures (ResNet, EfficientNet, VGG, Inception)  
- Real-world deep learning utilities and best practices  

Everything is implemented from scratch using PyTorch with clean and beginner-friendly code.

---

## Features

- Beginner → Advanced learning structure  
- Clean and well-commented PyTorch implementations  
- Real-world deep learning projects  
- CNN, RNN, LSTM, Transformers, GANs  
- Transfer learning & fine-tuning  
- Object detection and image segmentation  
- NLP projects including translation and text generation  
- Training utilities (TensorBoard, reproducibility, FP16, etc.)  
- Production-level project structure  

---

## Complete Topics with File Paths

### PyTorch Basics

1. [Initializing Tensors](tensor_1.py)  
2. [Tensor Math & Comparison Operations](tensor_2.py)  
3. [Tensor Indexing & Slicing](tensor_3.py)  
4. [Tensor Reshaping & Manipulation](tensor_4.py)  
5. [Build the First Sample Neural Network](nuralNetwork.py)  

---

### Deep Learning Models

6. [Build Basic CNN Model](CNN/cnn.py)  
7. [Build Basic RNN Model](RNN/rnn.py)  
8. [Using Bidirectional LSTM](RNN/BidirectionalLSTM.py)  
9. [Save and Load Model Checkpoints](CNN/loadSave.py)  
10. [Fine-Tuning & Transfer Learning using VGG16](TransferLearning+FineTuning/TransferLearningandFineTuning.py)  
11. [Import Data and Train on Pretrained Models](Build_Custom_Datasets/ImportData.py)  
12. [Custom Dataset for Image Captioning (Flickr8k)](Build_custom_textDatasets/text_data.py)  
13. [Image Data Augmentation using torchvision](Buid_Custom_Datasets/augmentation.py)  
14. [Albumentations Implementation](Albumentation/)  
15. [Handling Imbalanced Datasets](Imbalanced_classes/main.py)  
16. [TensorBoard Implementation on MNIST](Tensorboard/main.py)  

---

### CNN Architectures (From Scratch)

17. [LeNet Implementation](cnn_architectures/lenet/lenet5_pytorch.py)  
18. [VGG16 Implementation + General VGG Architecture](cnn_architectures/VGG/vgg_pytorch.py)  
19. [GoogLeNet / InceptionNet Implementation](cnn_architectures/InceptionNet/InceptionNet_pytorch.py)  
20. [ResNet Implementation](cnn_architectures/ResNet/ResNet_pytorch.py)  
21. [EfficientNet Implementation](cnn_architectures/EfficientNet/efficientNet_pytorch.py)  

---

### Computer Vision Projects

22. [Image Captioning from Scratch](Image_Captioning/)  
23. [Neural Style Transfer](nuralStyle/)  
24. [Simple GAN Implementation](GANs/SimpleGAN/simpleGAN.py)  
25. [DCGAN Implementation](GANs/DCGAN/)  
26. [Wasserstein GAN (WGAN)](GANs/WGAN/)  
27. [WGAN with Gradient Penalty (WGAN-GP)](GANs/WGAN-GP/)  
28. [Conditional GAN Implementation](GANs/ConditionalGAN/)  
29. [Pix2Pix (PatchGAN + U-Net)](GANs/Pix2Pix/)  
30. [CycleGAN (Horse ↔ Zebra)](GANs/CycleGAN/)  
31. [ProGAN Implementation](GANs/ProGAN/)  
32. [SRGAN – Super Resolution](GANs/SRGAN/)  
33. [ESRGAN – Enhanced Super Resolution](GANs/ESRGAN/)  

---

### NLP Projects

34. [Baby Name Generator using RNN-LSTM](TextGenerator/nameGenerator.py)  
35. [LSTM Text Classification using TorchText](torchText/part1/part1.py)  
36. [TorchText + SpaCy Data Pipeline (Multi30k)](torchText/part2/part2.py)  
37. [German ↔ English Translation Data Pipeline](torchText/part3/part3.py)  
38. [Seq2Seq Machine Translation (LSTM)](seq2seq/machineTranslation/)  
39. [Attention-Based Seq2Seq Model](seq2seq/machineTranslationWithAttention/)  
40. [Transformer Model from Scratch](Transformer/transformer.py)  
41. [Transformer-based Machine Translation](transformer_seq2seq/transformer_machinetranslation.py)  

---

### Image Segmentation & Object Detection

42. [U-Net Image Segmentation](UNET_imageSegmentation/)  
43. [Intersection over Union (IoU) Implementation](Object_detection/metrics/IoU.py)  
44. [Non-Maximum Suppression (NMS)](Object_detection/metrics/nms.py)  
45. [Mean Average Precision (mAP) Implementation](Object_detection/metrics/mAP.py)  
46. [YOLOv1 – End-to-End Object Detection](Object_detection/YOLOv1/)  
47. [Optimized YOLOv3 Implementation](Object_detection/YOLOv3/)  

---

### Quick PyTorch Tips & Utilities

48. [CNN with Mixed Precision Training (FP16)](QuickTips/fp16.py)  
49. [Training Progress Bar using tqdm](QuickTips/progress_bar.py)  
50. [Reproducibility – Fix Random Seeds](QuickTips/set_seeds.py)  
51. [Calculate Mean & Standard Deviation of Dataset](QuickTips/std_mean.py)  
52. [CNN with Proper Weight Initialization](QuickTips/weight_init.py)  
53. [Learning Rate Scheduling Implementation](QuickTips/lr_schedular.py)  

---

## Tech Stack

- Python  
- PyTorch  
- TorchText  
- torchvision  
- NumPy  
- TensorBoard  
- tqdm  
- Albumentations  

---

## Installation

```bash
git clone https://github.com/your-username/Pytorch-tutorial.git
cd Pytorch-tutorial
pip install -r requirements.txt

```
## Usage

Run any file directly
```bash
python CNN/cnn.py

```
## Author

Ravi Shankar Kumar  

If you found this repository helpful, consider giving it a star on GitHub.