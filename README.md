<h1 align="center">LitterCoast</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch">
  <img src="https://img.shields.io/badge/Computer%20Vision-Bounding%20Box%20Detection-green">
  <img src="https://img.shields.io/badge/Google%20Colab-Workflow-yellow">
</p>

## Overview

This repository currently focuses on a **bounding box detection pipeline** for coastal waste images.

Implemented components include:

- YOLOv8-based object detection
- Dataset download, extraction, and preparation for training
- Model training in Google Colab
- Model storage in Google Drive for later inference
- Image ingestion and result storage through Google Drive

## Detection Classes

The current bbox model is configured with 13 classes:

- pet_bottle
- other_bottle
- plastic_bag
- box_shaped_case
- other_container
- rope
- other_string
- fishing_net
- buoy
- other_fishing_gear
- styrene_foam
- others
- fragment

## Repository Files

- `yolo_bbox_training.py`: prepares the dataset structure and trains the YOLOv8 bbox model
- `bbox_image_inference.py`: loads images from Google Drive and stores prediction outputs in JSON
- `qr.py`: auxiliary QR code generation script

## Workflow

1. Images are collected through a Google Forms workflow.
2. Uploaded files are stored in Google Drive.
3. The dataset is downloaded from an external link as an archive file such as `.zip` or `.7z`.
4. In Google Colab, the dataset archive is extracted and prepared for training.
5. The YOLOv8 bbox model is trained.
6. The trained model file is saved to Google Drive.
7. Later, the saved model is loaded from Google Drive for inference.
8. Predictions are written back to Google Drive as output data.

## Execution Context

The current scripts are written around a **Google Colab + Google Drive** environment and use paths such as `/content/drive/...` and `/content/datasets/...`.

Before running the code, you should adapt:

- dataset download link
- dataset archive paths
- Google Drive mount steps
- trained model path in Drive
- output directories
- any local environment dependency handling
