# YOLOv1 Object Detection (PyTorch)

This project is a **clean and complete implementation of YOLOv1** using PyTorch.  
All files are already written – you only need to **copy the code into the correct files and run the training script**.

---

## What this project includes

This repository contains everything required to train a YOLOv1 model from scratch:

- YOLOv1 architecture (with BatchNorm)
- YOLO loss function (original paper implementation)
- Pascal VOC dataset loader
- IoU, NMS and mAP utilities
- Training script (CPU + GPU ready)

---

## Project Structure

Just create this folder structure and paste the code into the files.

--- YOLOv1/
│
├── model.py
├── loss.py
├── dataset.py
├── utils.py
├── train.py
│
├── data/
│ ├── images/
│ ├── labels/
│ ├── 100examples.csv
│ └── test.csv
│
└── README.md

## Dataset Format

Each image must have a matching `.txt` file.

Example:

Label format (YOLO format):

---

## How to Run

Step 1 – Install dependencies

Step 2 – Copy the code into the files

Step 3 – Add your dataset inside the `data/` folder

Step 4 – Run training

---

## Features

- Works on CPU
- Automatically uses GPU if available
- Clean and beginner-friendly code
- Good project for ML / AI portfolio

---

## Goal of this Project

This project is created for:

- Learning object detection
- Understanding YOLOv1
- Building a strong deep-learning project

---