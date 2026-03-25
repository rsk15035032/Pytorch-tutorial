# YOLOv3 Object Detection (PyTorch)

A clean and optimized implementation of **YOLOv3 from scratch using PyTorch**, trained on the **Pascal VOC dataset** with a focus on readability, performance, and real-world AI/ML project structure.

---

## Features

- YOLOv3 implemented completely from scratch  
- Multi-scale object detection (3 detection heads)  
- Custom YOLO loss function  
- Albumentations data augmentation pipeline  
- Non-Max Suppression (NMS) implementation  
- mAP (Mean Average Precision) evaluation  
- GPU + CPU compatible training  
- Clean and modular code structure  

---

## Project Structure
YOLOv3/
│
├── model.py
├── loss.py
├── dataset.py
├── utils.py
├── train.py
├── config.py
│
├── PASCAL_VOC/
│   ├── images/
│   ├── labels/
│   ├── train.csv
│   └── test.csv
│
└── README.md


---

## Dataset

This project uses the **Pascal VOC Dataset (20 classes)**:

person, car, dog, cat, bus, bicycle, horse, bottle, chair, train, and more.

---

## Installation

git clone https://github.com/rsk15035032/Pytorch-tutorial/tree/main/Object_detection/YOLOv3
cd yolov3-pytorch
pip install -r requirements.txt
python train.py


The model supports:

- GPU training (CUDA)  
- CPU training (low-RAM laptops)  
- Checkpoint saving & loading  
- mAP evaluation every few epochs  

---

## Results

- Multi-scale predictions (13×13, 26×26, 52×52)  
- Stable training with custom YOLO loss  
- Clean bounding box predictions after NMS  

---

## Tech Stack

- Python  
- PyTorch  
- Albumentations  
- NumPy  
- OpenCV  
- Matplotlib  

---

## Purpose of this Project

This project is built as part of an **AI/ML Engineer portfolio** to demonstrate:

- Deep understanding of object detection  
- Training pipelines in PyTorch  
- Clean and production-level code  

---

## Author

**Ravi Shankar Kumar**  
ML Engineer
