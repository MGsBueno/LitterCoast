from __future__ import annotations

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


def _resolve_request_environment(environment: AppEnvironment | None) -> AppEnvironment:
    return environment or get_default_environment()


def _resolve_path_within_roots(path: Path, allowed_roots: tuple[Path, ...]) -> Path:
    resolved_path = path.resolve()

    for allowed_root in allowed_roots:
        resolved_root = allowed_root.resolve()
        try:
            if resolved_path.is_relative_to(resolved_root):
                return resolved_path
        except AttributeError:
            if str(resolved_path).startswith(str(resolved_root)):
                return resolved_path

    allowed_roots_message = ", ".join(str(root) for root in allowed_roots)
    raise ValueError(f"path '{path}' must stay within allowed roots: {allowed_roots_message}")


def _get_allowed_roots(environment: AppEnvironment) -> dict[str, tuple[Path, ...]]:
    if environment == AppEnvironment.LOCAL:
        return {
            "dataset_archive_path": (Path.cwd() / "data",),
            "extracted_dir": (Path.cwd() / "data",),
            "dataset_root": (Path.cwd() / "data",),
            "dataset_yaml_path": (Path.cwd() / "data",),
            "model_path": (Path.cwd() / "models",),
            "image_directory": (Path.cwd() / "data",),
            "predictions_path": (Path.cwd() / "outputs",),
            "output_path": (Path.cwd() / "outputs",),
        }

    if environment == AppEnvironment.COLAB:
        return {
            "dataset_archive_path": (Path("/content"), Path("/content/drive")),
            "extracted_dir": (Path("/content"),),
            "dataset_root": (Path("/content"),),
            "dataset_yaml_path": (Path("/content"),),
            "model_path": (Path("/content"), Path("/content/drive")),
            "image_directory": (Path("/content"), Path("/content/drive")),
            "predictions_path": (Path("/content"), Path("/content/drive")),
            "output_path": (Path("/content"),),
        }

    training_config = TrainingConfig.for_environment(environment)
    inference_config = InferenceConfig.for_environment(environment)
    qr_config = QRCodeConfig.for_environment(environment)
    return {
        "dataset_archive_path": (training_config.dataset_archive_path.parent,),
        "extracted_dir": (training_config.extracted_dir.parent,),
        "dataset_root": (training_config.dataset_root.parent,),
        "dataset_yaml_path": (training_config.dataset_yaml_path.parent,),
        "model_path": (inference_config.model_path.parent,),
        "image_directory": (inference_config.image_directory.parent,),
        "predictions_path": (inference_config.predictions_path.parent,),
        "output_path": (qr_config.output_path.parent,),
    }


def _build_training_config(request: TrainingRequest, environment: AppEnvironment) -> TrainingConfig:
    config = TrainingConfig.for_environment(environment)
    allowed_roots = _get_allowed_roots(environment)
    if request.dataset_archive_path:
        config.dataset_archive_path = _resolve_path_within_roots(Path(request.dataset_archive_path), allowed_roots["dataset_archive_path"])
    if request.extracted_dir:
        config.extracted_dir = _resolve_path_within_roots(Path(request.extracted_dir), allowed_roots["extracted_dir"])
    if request.dataset_root:
        config.dataset_root = _resolve_path_within_roots(Path(request.dataset_root), allowed_roots["dataset_root"])
    if request.dataset_yaml_path:
        config.dataset_yaml_path = _resolve_path_within_roots(Path(request.dataset_yaml_path), allowed_roots["dataset_yaml_path"])
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


def _build_inference_config(request: InferenceRequest, environment: AppEnvironment) -> InferenceConfig:
    config = InferenceConfig.for_environment(environment)
    allowed_roots = _get_allowed_roots(environment)
    if request.model_path:
        config.model_path = _resolve_path_within_roots(Path(request.model_path), allowed_roots["model_path"])
    if request.image_directory:
        config.image_directory = _resolve_path_within_roots(Path(request.image_directory), allowed_roots["image_directory"])
    if request.predictions_path:
        config.predictions_path = _resolve_path_within_roots(Path(request.predictions_path), allowed_roots["predictions_path"])
    return config


def _build_qr_config(request: QRCodeRequest, environment: AppEnvironment) -> QRCodeConfig:
    config = QRCodeConfig.for_environment(environment)
    allowed_roots = _get_allowed_roots(environment)
    if request.link:
        config.link = request.link
    if request.output_path:
        config.output_path = _resolve_path_within_roots(Path(request.output_path), allowed_roots["output_path"])
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
        environment = _resolve_request_environment(request.environment)
        config = _run_with_http_error_handling(
            lambda: _build_training_config(request, environment),
            "training configuration failed",
        )
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
        environment = _resolve_request_environment(request.environment)
        config = _run_with_http_error_handling(
            lambda: _build_inference_config(request, environment),
            "inference configuration failed",
        )
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
        environment = _resolve_request_environment(request.environment)
        config = _run_with_http_error_handling(
            lambda: _build_qr_config(request, environment),
            "qr configuration failed",
        )
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
