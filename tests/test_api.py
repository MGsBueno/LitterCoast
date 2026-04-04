from pathlib import Path
import os

import pytest

from littercoast.api import (
    _build_inference_config,
    _build_qr_config,
    _build_training_config,
    _resolve_path_within_roots,
    _resolve_request_environment,
    app,
)
from littercoast.config import AppEnvironment
from littercoast.schemas import InferenceRequest, QRCodeRequest, TrainingRequest


def _get_route_endpoint(path: str, method: str):
    for route in app.routes:
        if route["path"] == path and route["method"] == method:
            return route["endpoint"]
    raise AssertionError(f"route not found: {method} {path}")


@pytest.mark.unit
def test_resolve_request_environment_does_not_mutate_process_environment(monkeypatch) -> None:
    original_environment = "colab"
    monkeypatch.setenv("LITTERCOAST_ENVIRONMENT", original_environment)

    result = _resolve_request_environment(AppEnvironment.LOCAL)

    assert result == AppEnvironment.LOCAL
    assert os.environ["LITTERCOAST_ENVIRONMENT"] == original_environment


@pytest.mark.unit
def test_resolve_request_environment_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("LITTERCOAST_ENVIRONMENT", raising=False)

    result = _resolve_request_environment(AppEnvironment.LOCAL)

    assert result == AppEnvironment.LOCAL


@pytest.mark.unit
def test_build_training_config_applies_request_overrides(monkeypatch) -> None:
    monkeypatch.delenv("LITTERCOAST_DATASET_ARCHIVE_PATH", raising=False)
    monkeypatch.delenv("LITTERCOAST_DATASET_ROOT", raising=False)
    request = TrainingRequest(
        environment=AppEnvironment.LOCAL,
        dataset_archive_path="./data/dataset.zip",
        dataset_root="./data/custom-dataset-root",
        epochs=3,
        run_name="api-train",
    )

    config = _build_training_config(request, AppEnvironment.LOCAL)

    assert config.dataset_archive_path == (Path.cwd() / "data" / "dataset.zip").resolve()
    assert config.dataset_root == (Path.cwd() / "data" / "custom-dataset-root").resolve()
    assert config.epochs == 3
    assert config.run_name == "api-train"


@pytest.mark.unit
def test_build_inference_config_applies_request_overrides(monkeypatch) -> None:
    monkeypatch.delenv("LITTERCOAST_MODEL_PATH", raising=False)
    request = InferenceRequest(
        environment=AppEnvironment.LOCAL,
        model_path="./models/custom.pt",
        image_directory="./data/images",
        predictions_path="./outputs/predictions.json",
    )

    config = _build_inference_config(request, AppEnvironment.LOCAL)

    assert config.model_path == (Path.cwd() / "models" / "custom.pt").resolve()
    assert config.image_directory == (Path.cwd() / "data" / "images").resolve()
    assert config.predictions_path == (Path.cwd() / "outputs" / "predictions.json").resolve()


@pytest.mark.unit
def test_build_qr_config_applies_request_overrides() -> None:
    request = QRCodeRequest(
        environment=AppEnvironment.LOCAL,
        link="https://example.com",
        output_path="./outputs/qr.png",
        version=2,
        box_size=6,
        border=1,
    )

    config = _build_qr_config(request, AppEnvironment.LOCAL)

    assert config.link == "https://example.com"
    assert config.output_path == (Path.cwd() / "outputs" / "qr.png").resolve()
    assert config.version == 2
    assert config.box_size == 6
    assert config.border == 1


@pytest.mark.unit
def test_resolve_path_within_roots_blocks_path_traversal() -> None:
    allowed_root = Path.cwd() / "data"

    with pytest.raises(ValueError, match="must stay within allowed roots"):
        _resolve_path_within_roots(Path("../secret.txt"), (allowed_root,))


@pytest.mark.unit
def test_api_registers_expected_routes() -> None:
    route_map = {(route["method"], route["path"]) for route in app.routes}

    assert ("GET", "/health") in route_map
    assert ("GET", "/config/environment") in route_map
    assert ("POST", "/train") in route_map
    assert ("POST", "/infer") in route_map
    assert ("POST", "/qr") in route_map


@pytest.mark.unit
def test_health_endpoint_returns_ok_message() -> None:
    endpoint = _get_route_endpoint("/health", "GET")

    response = endpoint()

    assert response.message == "ok"


@pytest.mark.unit
def test_train_endpoint_runs_trainer_and_returns_summary(monkeypatch) -> None:
    endpoint = _get_route_endpoint("/train", "POST")
    captured = {}

    class FakeTrainer:
        def __init__(self, training_config):
            captured["config"] = training_config

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("littercoast.training.ModelTrainer", FakeTrainer)

    response = endpoint(
        TrainingRequest(
            environment=AppEnvironment.LOCAL,
            dataset_root="./data/train-root",
            dataset_yaml_path="./data/waste.yaml",
            run_name="api-run",
        )
    )

    assert captured["ran"] is True
    assert captured["config"].dataset_root == (Path.cwd() / "data" / "train-root").resolve()
    assert response.message == "training finished"
    assert response.environment == AppEnvironment.LOCAL
    assert response.run_name == "api-run"


@pytest.mark.unit
def test_infer_endpoint_runs_pipeline_and_returns_summary(monkeypatch) -> None:
    endpoint = _get_route_endpoint("/infer", "POST")
    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return [{"image": "a.jpg"}, {"image": "b.jpg"}]

    monkeypatch.setattr("littercoast.inference.InferencePipeline", FakePipeline)

    response = endpoint(
        InferenceRequest(
            environment=AppEnvironment.LOCAL,
            predictions_path="./outputs/result.json",
        )
    )

    assert captured["config"].predictions_path == (Path.cwd() / "outputs" / "result.json").resolve()
    assert response.message == "inference finished"
    assert response.predictions_count == 2


@pytest.mark.unit
def test_qr_endpoint_runs_generator_and_returns_summary(monkeypatch) -> None:
    endpoint = _get_route_endpoint("/qr", "POST")
    captured = {}

    class FakeGenerator:
        def __init__(self, config):
            captured["config"] = config

        def generate(self):
            return captured["config"].output_path

    monkeypatch.setattr("littercoast.qr_code.QRCodeGenerator", FakeGenerator)

    response = endpoint(
        QRCodeRequest(
            environment=AppEnvironment.LOCAL,
            output_path="./outputs/generated-qr.png",
        )
    )

    assert captured["config"].output_path == (Path.cwd() / "outputs" / "generated-qr.png").resolve()
    assert response.message == "qr code generated"
    assert Path(response.output_path) == (Path.cwd() / "outputs" / "generated-qr.png").resolve()
