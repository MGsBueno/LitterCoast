from pathlib import Path

import pytest

from littercoast.api import (
    _apply_environment,
    _build_inference_config,
    _build_qr_config,
    _build_training_config,
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
def test_apply_environment_updates_runtime_environment(monkeypatch) -> None:
    monkeypatch.delenv("LITTERCOAST_ENVIRONMENT", raising=False)

    result = _apply_environment(AppEnvironment.LOCAL)

    assert result == AppEnvironment.LOCAL


@pytest.mark.unit
def test_build_training_config_applies_request_overrides(monkeypatch) -> None:
    monkeypatch.delenv("LITTERCOAST_DATASET_ARCHIVE_PATH", raising=False)
    monkeypatch.delenv("LITTERCOAST_DATASET_ROOT", raising=False)
    request = TrainingRequest(
        environment=AppEnvironment.LOCAL,
        dataset_archive_path="./custom/dataset.zip",
        dataset_root="./custom/dataset-root",
        epochs=3,
        run_name="api-train",
    )

    config = _build_training_config(request)

    assert config.dataset_archive_path == Path("custom/dataset.zip")
    assert config.dataset_root == Path("custom/dataset-root")
    assert config.epochs == 3
    assert config.run_name == "api-train"


@pytest.mark.unit
def test_build_inference_config_applies_request_overrides(monkeypatch) -> None:
    monkeypatch.delenv("LITTERCOAST_MODEL_PATH", raising=False)
    request = InferenceRequest(
        environment=AppEnvironment.LOCAL,
        model_path="./models/custom.pt",
        image_directory="./images",
        predictions_path="./outputs/predictions.json",
    )

    config = _build_inference_config(request)

    assert config.model_path == Path("models/custom.pt")
    assert config.image_directory == Path("images")
    assert config.predictions_path == Path("outputs/predictions.json")


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

    config = _build_qr_config(request)

    assert config.link == "https://example.com"
    assert config.output_path == Path("outputs/qr.png")
    assert config.version == 2
    assert config.box_size == 6
    assert config.border == 1


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
    assert captured["config"].dataset_root == Path("data/train-root")
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

    assert captured["config"].predictions_path == Path("outputs/result.json")
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
            return Path("./outputs/generated-qr.png")

    monkeypatch.setattr("littercoast.qr_code.QRCodeGenerator", FakeGenerator)

    response = endpoint(
        QRCodeRequest(
            environment=AppEnvironment.LOCAL,
            output_path="./outputs/generated-qr.png",
        )
    )

    assert captured["config"].output_path == Path("outputs/generated-qr.png")
    assert response.message == "qr code generated"
    assert Path(response.output_path) == Path("outputs/generated-qr.png")
