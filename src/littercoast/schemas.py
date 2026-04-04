from __future__ import annotations

from pydantic import BaseModel, Field

from .config import AppEnvironment


class TrainingRequest(BaseModel):
    environment: AppEnvironment | None = None
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
    environment: AppEnvironment | None = None
    model_path: str | None = None
    image_directory: str | None = None
    predictions_path: str | None = None


class QRCodeRequest(BaseModel):
    environment: AppEnvironment | None = None
    link: str | None = None
    output_path: str | None = None
    version: int | None = Field(default=None, ge=1)
    box_size: int | None = Field(default=None, ge=1)
    border: int | None = Field(default=None, ge=0)


class EnvironmentResponse(BaseModel):
    environment: AppEnvironment


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str


class TrainingResponse(BaseModel):
    message: str
    environment: AppEnvironment
    dataset_root: str
    dataset_yaml_path: str
    run_name: str


class InferenceResponse(BaseModel):
    message: str
    environment: AppEnvironment
    predictions_path: str
    predictions_count: int


class QRCodeResponse(BaseModel):
    message: str
    environment: AppEnvironment
    output_path: str
