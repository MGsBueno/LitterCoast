from __future__ import annotations

from pydantic import BaseModel, Field

from .config import AppEnvironment


class TrainingRequest(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "environment": "local",
                "dataset_archive_path": "./data/dataset.tar.gz",
                "extracted_dir": "./data/extracted/beach_plastic_litter_dataset_v2",
                "dataset_root": "./data/garbage_classification",
                "dataset_yaml_path": "./data/waste.yaml",
                "model_name": "yolov8n.pt",
                "epochs": 10,
            }
        }

    environment: AppEnvironment | None = Field(default=None, examples=["local"])
    dataset_archive_path: str | None = None
    extracted_dir: str | None = None
    dataset_root: str | None = None
    dataset_yaml_path: str | None = None
    model_name: str | None = None
    epochs: int | None = Field(default=None, ge=1)
    image_size: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    run_name: str | None = None


class InferenceRequest(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "environment": "local",
                "model_path": "./models/yolov8_model.pt",
                "image_directory": "./data/images",
                "predictions_path": "./outputs/yolo_predictions.json",
            }
        }

    environment: AppEnvironment | None = Field(default=None, examples=["local"])
    model_path: str | None = None
    image_directory: str | None = None
    predictions_path: str | None = None


class QRCodeRequest(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "environment": "local",
                "link": "https://forms.gle/PLDgtbQkSxKboPfv7",
                "output_path": "./outputs/qrcode_link.png",
            }
        }

    environment: AppEnvironment | None = Field(default=None, examples=["local"])
    link: str | None = None
    output_path: str | None = None
    version: int | None = Field(default=None, ge=1)
    box_size: int | None = Field(default=None, ge=1)
    border: int | None = Field(default=None, ge=0)


class EnvironmentResponse(BaseModel):
    class Config:
        schema_extra = {"example": {"environment": "local"}}

    environment: AppEnvironment = Field(examples=["local"])


class MessageResponse(BaseModel):
    class Config:
        schema_extra = {"example": {"message": "ok"}}

    message: str


class ErrorResponse(BaseModel):
    detail: str


class TrainingResponse(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "message": "training finished",
                "environment": "local",
                "dataset_root": "/absolute/path/to/project/data/garbage_classification",
                "dataset_yaml_path": "/absolute/path/to/project/data/waste.yaml",
                "run_name": "garbage_detection_yolov8",
            }
        }

    message: str
    environment: AppEnvironment = Field(examples=["local"])
    dataset_root: str
    dataset_yaml_path: str
    run_name: str


class InferenceResponse(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "message": "inference finished",
                "environment": "local",
                "predictions_path": "/absolute/path/to/project/outputs/yolo_predictions.json",
                "predictions_count": 12,
            }
        }

    message: str
    environment: AppEnvironment = Field(examples=["local"])
    predictions_path: str
    predictions_count: int


class QRCodeResponse(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "message": "qr code generated",
                "environment": "local",
                "output_path": "/absolute/path/to/project/outputs/qrcode_link.png",
            }
        }

    message: str
    environment: AppEnvironment = Field(examples=["local"])
    output_path: str
