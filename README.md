  <h1 align="center">LitterCoast</h1>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
    <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange">
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch">
    <img src="https://img.shields.io/badge/Computer%20Vision-Object%20Detection-green">
    <img src="https://img.shields.io/badge/IEEE-OES-blue">
  </p>

  LitterCoast is a computer vision project focused on detecting improperly discarded waste in coastal environments using deep learning.

  ## Overview

  The current version of the project focuses on building and testing an **image detection pipeline** capable of identifying marine litter from images.

  Implemented components include:

  - Waste detection using **YOLOv8**
  - Model training and evaluation using the **[Sea Computer Vision Project dataset](https://universe.roboflow.com/hongmo/sea-ezx3q)**
  - Image storage workflow using **Google Drive**
  - Collaborative image submission via **Google Forms**

  ---

  ## Detection Classes

  The model currently detects the following waste categories:

  - Glass
  - Metal
  - Net
  - PET Bottle
  - Plastic Buoy
  - Plastic Buoy China
  - Plastic ETC
  - Rope
  - Styrofoam Box
  - Styrofoam Buoy
  - Styrofoam Piece

  These classes were selected based on materials commonly found in marine litter datasets.

  ---

  ## Pipeline

  The current workflow of the project is shown below:

  1. Users upload images through **Google Forms**
  2. Images are stored in **Google Drive**
  3. A **Google Colab notebook** retrieves the images
  4. The **YOLOv8 model performs detection**
  5. Processed images and results are saved back to Google Drive

  This architecture allows easy experimentation without requiring dedicated infrastructure.
