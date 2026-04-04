from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .config import AppEnvironment, InferenceConfig, QRCodeConfig, TrainingConfig, get_default_environment
from .schemas import (
    EnvironmentResponse,
    ErrorResponse,
    InferenceRequest,
    InferenceResponse,
    MessageResponse,
    QRCodeRequest,
    QRCodeResponse,
    TrainingRequest,
    TrainingResponse,
)


def _apply_environment(environment: AppEnvironment | None) -> AppEnvironment:
    if environment is not None:
        os.environ["LITTERCOAST_ENVIRONMENT"] = environment.value
    return get_default_environment()


def _build_training_config(request: TrainingRequest) -> TrainingConfig:
    config = TrainingConfig()
    if request.dataset_archive_path:
        config.dataset_archive_path = Path(request.dataset_archive_path)
    if request.extracted_dir:
        config.extracted_dir = Path(request.extracted_dir)
    if request.dataset_root:
        config.dataset_root = Path(request.dataset_root)
    if request.dataset_yaml_path:
        config.dataset_yaml_path = Path(request.dataset_yaml_path)
    if request.model_name:
        config.model_name = request.model_name
    if request.epochs is not None:
        config.epochs = request.epochs
    if request.image_size is not None:
        config.image_size = request.image_size
    if request.batch_size is not None:
        config.batch_size = request.batch_size
    if request.run_name:
        config.run_name = request.run_name
    return config


def _build_inference_config(request: InferenceRequest) -> InferenceConfig:
    config = InferenceConfig()
    if request.model_path:
        config.model_path = Path(request.model_path)
    if request.image_directory:
        config.image_directory = Path(request.image_directory)
    if request.predictions_path:
        config.predictions_path = Path(request.predictions_path)
    return config


def _build_qr_config(request: QRCodeRequest) -> QRCodeConfig:
    config = QRCodeConfig()
    if request.link:
        config.link = request.link
    if request.output_path:
        config.output_path = Path(request.output_path)
    if request.version is not None:
        config.version = request.version
    if request.box_size is not None:
        config.box_size = request.box_size
    if request.border is not None:
        config.border = request.border
    return config


def _run_with_http_error_handling(action, failure_message: str):
    try:
        return action()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{failure_message}: {exc}") from exc


def create_app() -> FastAPI:
    app = FastAPI(
        title="LitterCoast API",
        version="0.2.0b0",
        description="API for LitterCoast training, inference, QR generation, and environment inspection.",
    )

    @app.get("/health", response_model=MessageResponse)
    def healthcheck() -> MessageResponse:
        return MessageResponse(message="ok")

    @app.get("/config/environment", response_model=EnvironmentResponse)
    def get_environment() -> EnvironmentResponse:
        return EnvironmentResponse(environment=get_default_environment())

    @app.post(
        "/train",
        response_model=TrainingResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def train_model(request: TrainingRequest) -> TrainingResponse:
        environment = _apply_environment(request.environment)
        config = _build_training_config(request)
        _run_with_http_error_handling(
            lambda: __import__("littercoast.training", fromlist=["ModelTrainer"]).ModelTrainer(training_config=config).run(),
            "training failed",
        )
        return TrainingResponse(
            message="training finished",
            environment=environment,
            dataset_root=str(config.dataset_root),
            dataset_yaml_path=str(config.dataset_yaml_path),
            run_name=config.run_name,
        )

    @app.post(
        "/infer",
        response_model=InferenceResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def run_inference(request: InferenceRequest) -> InferenceResponse:
        environment = _apply_environment(request.environment)
        config = _build_inference_config(request)
        predictions = _run_with_http_error_handling(
            lambda: __import__("littercoast.inference", fromlist=["InferencePipeline"]).InferencePipeline(config).run(),
            "inference failed",
        )
        return InferenceResponse(
            message="inference finished",
            environment=environment,
            predictions_path=str(config.predictions_path),
            predictions_count=len(predictions),
        )

    @app.post(
        "/qr",
        response_model=QRCodeResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def generate_qr_code(request: QRCodeRequest) -> QRCodeResponse:
        environment = _apply_environment(request.environment)
        config = _build_qr_config(request)
        output_path = _run_with_http_error_handling(
            lambda: __import__("littercoast.qr_code", fromlist=["QRCodeGenerator"]).QRCodeGenerator(config).generate(),
            "qr generation failed",
        )
        return QRCodeResponse(
            message="qr code generated",
            environment=environment,
            output_path=str(output_path),
        )

    return app


app = create_app()
