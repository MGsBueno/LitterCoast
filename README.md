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

LitterCoast evolved from an experimental Colab-based prototype into a modular backend application with CLI, API, environment-aware configuration, and automated tests.

The project was presented in the IEEE OES Ocean Challenge in 2025 as part of its early evolution.

## Product Direction

This repository is currently backend-first.

The priority is to consolidate:

- training and inference services
- API contracts
- environment configuration
- test coverage
- backend documentation

The frontend is intentionally a later step and should consume the API instead of coupling UI logic directly to the training scripts.

## Current Architecture

```text
LitterCoast/
|-- src/
|   `-- littercoast/
|       |-- api.py
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli.py
|       |-- config.py
|       |-- inference.py
|       |-- qr_code.py
|       |-- schemas.py
|       `-- training.py
|-- bbox_image_inference.py
|-- qr.py
|-- tests/
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
pip install -e .[dev]
```

Create your environment file:

```bash
cp .env.example .env
```

All default paths and runtime parameters can now be overridden through `.env`.

You can define the runtime preset with `LITTERCOAST_ENVIRONMENT`:

- `colab`: paths aimed at Google Colab + Drive
- `local`: paths aimed at local folders such as `./data`, `./models`, and `./outputs`
- `custom`: keeps the explicit values from `.env` as the source of truth

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

You can also pick the setup directly in the command:

```bash
python -m littercoast --environment colab train
python -m littercoast --environment local infer
python -m littercoast --environment local qr
python -m littercoast --environment local api --host 0.0.0.0 --port 8000
```

## Running Tests

Run all unit tests:

```bash
pytest -q tests -m unit
```

Run all integration tests:

```bash
pytest -q tests -m integration
```

Run both suites together:

```bash
pytest -q tests -m "unit or integration"
```

## API

The project now includes a FastAPI backend so the frontend can be developed independently.

Main endpoints:

- `GET /health`
- `GET /config/environment`
- `POST /train`
- `POST /infer`
- `POST /qr`

Run the API with:

```bash
python -m littercoast api
```

Run in development mode:

```bash
python -m littercoast --environment local api --host 0.0.0.0 --port 8000
```

Interactive documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

In `local` mode, API-managed paths are restricted to project-safe roots:

- `./data`
- `./models`
- `./outputs`

The API validates these inputs and resolves accepted local paths to absolute paths in responses.

### Endpoints

`GET /health`

Response example:

```json
{
  "message": "ok"
}
```

`GET /config/environment`

Response example:

```json
{
  "environment": "local"
}
```

`POST /train`

Request body example:

```json
{
  "environment": "local",
  "dataset_archive_path": "./data/dataset.tar.gz",
  "extracted_dir": "./data/extracted/beach_plastic_litter_dataset_v2",
  "dataset_root": "./data/garbage_classification",
  "dataset_yaml_path": "./data/waste.yaml",
  "model_name": "yolov8n.pt",
  "epochs": 10
}
```

Response example:

```json
{
  "message": "training finished",
  "environment": "local",
  "dataset_root": "/absolute/path/to/project/data/garbage_classification",
  "dataset_yaml_path": "/absolute/path/to/project/data/waste.yaml",
  "run_name": "garbage_detection_yolov8"
}
```

`POST /infer`

Request body example:

```json
{
  "environment": "local",
  "model_path": "./models/yolov8_model.pt",
  "image_directory": "./data/images",
  "predictions_path": "./outputs/yolo_predictions.json"
}
```

Response example:

```json
{
  "message": "inference finished",
  "environment": "local",
  "predictions_path": "/absolute/path/to/project/outputs/yolo_predictions.json",
  "predictions_count": 12
}
```

`POST /qr`

Request body example:

```json
{
  "environment": "local",
  "link": "https://forms.gle/PLDgtbQkSxKboPfv7",
  "output_path": "./outputs/qrcode_link.png"
}
```

Response example:

```json
{
  "message": "qr code generated",
  "environment": "local",
  "output_path": "/absolute/path/to/project/outputs/qrcode_link.png"
}
```

### cURL Examples

Healthcheck:

```bash
curl http://localhost:8000/health
```

Training:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d "{\"environment\":\"local\",\"dataset_archive_path\":\"./data/dataset.tar.gz\",\"dataset_root\":\"./data/garbage_classification\",\"epochs\":10}"
```

Inference:

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d "{\"environment\":\"local\",\"model_path\":\"./models/yolov8_model.pt\",\"image_directory\":\"./data/images\",\"predictions_path\":\"./outputs/yolo_predictions.json\"}"
```

QR generation:

```bash
curl -X POST http://localhost:8000/qr \
  -H "Content-Type: application/json" \
  -d "{\"environment\":\"local\",\"link\":\"https://forms.gle/PLDgtbQkSxKboPfv7\",\"output_path\":\"./outputs/qrcode_link.png\"}"
```

### Current API Scope

The API currently focuses on orchestration of backend services.

Already covered:

- environment selection
- training orchestration
- inference orchestration
- QR generation
- typed request and response schemas
- local path validation for API-managed filesystem operations

Examples with custom paths:

```bash
python -m littercoast train --archive /content/drive/MyDrive/dataset.tar.gz --dataset-root /content/datasets/garbage_classification
python -m littercoast infer --model /content/drive/My\ Drive/yolov8_model.pt --images /content/drive/My\ Drive/images --output /content/drive/My\ Drive/yolo_predictions.json
python -m littercoast qr --link https://forms.gle/PLDgtbQkSxKboPfv7 --output qrcode_link.png
```

## Next Steps

- Introduce structured logging instead of `print`.
- Expand integration coverage for real HTTP flows and failure scenarios with the real FastAPI test client.
- Evolve training into asynchronous execution with status tracking.
- Persist job history and inference metadata.
- Add authentication, authorization, CORS, and production hardening.
- Add real database persistence for API-managed resources and history.
- Build the frontend on top of the API while keeping the backend-first strategy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
