<h1 align="center">LitterCoast</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch">
  <img src="https://img.shields.io/badge/Architecture-OO%20%2B%20Modular-success">
</p>

## Overview

LitterCoast is a coastal waste detection project built around a YOLOv8 pipeline.

The repository now follows a more modular and object-oriented structure so training, inference, and QR code generation can evolve independently without concentrating all logic in standalone scripts.

## Current Architecture

```text
LitterCoast/
|-- src/
|   `-- littercoast/
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli.py
|       |-- config.py
|       |-- inference.py
|       |-- qr_code.py
|       `-- training.py
|-- bbox_image_inference.py
|-- qr.py
|-- yolo_bbox_training.py
|-- pyproject.toml
`-- README.md
```

## OO Improvements Applied

- `TrainingConfig`, `InferenceConfig`, and `QRCodeConfig` centralize configuration.
- `ArchiveExtractor`, `DatasetPreparer`, and `ModelTrainer` separate dataset extraction, preparation, and training responsibilities.
- `ObjectDetector`, `PredictionRepository`, and `InferencePipeline` split inference into detection, persistence, and orchestration.
- `QRCodeGenerator` isolates QR creation logic.
- Legacy scripts were preserved as thin entrypoints that reuse the package code.

## Dataset

The current dataset source remains the Sea Computer Vision Project on Roboflow:

- https://universe.roboflow.com/hongmo/sea-ezx3q

Configured classes:

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

## How To Run

Install the package dependencies:

```bash
pip install -e .
```

Run with the preserved scripts:

```bash
python yolo_bbox_training.py
python bbox_image_inference.py
python qr.py
```

Or use the package CLI:

```bash
python -m littercoast train
python -m littercoast infer
python -m littercoast qr
```

Examples with custom paths:

```bash
python -m littercoast train --archive /content/drive/MyDrive/dataset.tar.gz --dataset-root /content/datasets/garbage_classification
python -m littercoast infer --model /content/drive/My\ Drive/yolov8_model.pt --images /content/drive/My\ Drive/images --output /content/drive/My\ Drive/yolo_predictions.json
python -m littercoast qr --link https://forms.gle/PLDgtbQkSxKboPfv7 --output qrcode_link.png
```

## Next Structural Suggestions

- Add automated tests for `DatasetPreparer` and `PredictionRepository`.
- Move environment-specific paths to `.env` or a config file.
- Introduce logging instead of `print`.
- Add a dedicated `tests/` directory and CI validation.
