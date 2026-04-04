from pathlib import Path

import pytest

from littercoast.config import AppEnvironment


@pytest.mark.integration
def test_health_get_returns_ok(api_client) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


@pytest.mark.integration
def test_environment_get_returns_active_environment(monkeypatch, api_client) -> None:
    monkeypatch.setenv("LITTERCOAST_ENVIRONMENT", AppEnvironment.LOCAL.value)

    response = api_client.get("/config/environment")

    assert response.status_code == 200
    assert response.json() == {"environment": "local"}


@pytest.mark.integration
def test_train_post_returns_training_summary(monkeypatch, api_client) -> None:
    captured = {}

    class FakeTrainer:
        def __init__(self, training_config):
            captured["config"] = training_config

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("littercoast.training.ModelTrainer", FakeTrainer)

    response = api_client.post(
        "/train",
        json={
            "environment": "local",
            "dataset_root": "./data/train-root",
            "dataset_yaml_path": "./data/waste.yaml",
            "run_name": "integration-train",
        },
    )

    assert response.status_code == 200
    assert captured["ran"] is True
    assert captured["config"].dataset_root == Path("data/train-root")
    assert response.json() == {
        "message": "training finished",
        "environment": "local",
        "dataset_root": str(Path("data/train-root")),
        "dataset_yaml_path": str(Path("data/waste.yaml")),
        "run_name": "integration-train",
    }


@pytest.mark.integration
def test_infer_post_returns_inference_summary(monkeypatch, api_client) -> None:
    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return [{"image": "sample-a.jpg"}, {"image": "sample-b.jpg"}]

    monkeypatch.setattr("littercoast.inference.InferencePipeline", FakePipeline)

    response = api_client.post(
        "/infer",
        json={
            "environment": "local",
            "predictions_path": "./outputs/predictions.json",
        },
    )

    assert response.status_code == 200
    assert captured["config"].predictions_path == Path("outputs/predictions.json")
    assert response.json() == {
        "message": "inference finished",
        "environment": "local",
        "predictions_path": str(Path("outputs/predictions.json")),
        "predictions_count": 2,
    }


@pytest.mark.integration
def test_qr_post_returns_qr_summary(monkeypatch, api_client) -> None:
    captured = {}

    class FakeGenerator:
        def __init__(self, config):
            captured["config"] = config

        def generate(self):
            return Path("./outputs/generated-qr.png")

    monkeypatch.setattr("littercoast.qr_code.QRCodeGenerator", FakeGenerator)

    response = api_client.post(
        "/qr",
        json={
            "environment": "local",
            "link": "https://example.com/form",
            "output_path": "./outputs/generated-qr.png",
        },
    )

    assert response.status_code == 200
    assert captured["config"].output_path == Path("outputs/generated-qr.png")
    assert response.json() == {
        "message": "qr code generated",
        "environment": "local",
        "output_path": str(Path("outputs/generated-qr.png")),
    }


@pytest.mark.integration
def test_infer_post_returns_not_found_for_missing_resource(monkeypatch, api_client) -> None:
    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            raise FileNotFoundError("model file not found")

    monkeypatch.setattr("littercoast.inference.InferencePipeline", FakePipeline)

    response = api_client.post("/infer", json={"environment": "local"})

    assert response.status_code == 404
    assert response.json() == {"detail": "model file not found"}


@pytest.mark.integration
def test_train_post_returns_bad_request_for_invalid_archive(monkeypatch, api_client) -> None:
    class FakeTrainer:
        def __init__(self, training_config):
            self.training_config = training_config

        def run(self):
            raise ValueError("unsupported archive format")

    monkeypatch.setattr("littercoast.training.ModelTrainer", FakeTrainer)

    response = api_client.post("/train", json={"environment": "local"})

    assert response.status_code == 400
    assert response.json() == {"detail": "unsupported archive format"}


@pytest.mark.integration
def test_qr_post_returns_internal_error_for_unexpected_failure(monkeypatch, api_client) -> None:
    class FakeGenerator:
        def __init__(self, config):
            self.config = config

        def generate(self):
            raise RuntimeError("disk write failed")

    monkeypatch.setattr("littercoast.qr_code.QRCodeGenerator", FakeGenerator)

    response = api_client.post("/qr", json={"environment": "local"})

    assert response.status_code == 500
    assert response.json() == {"detail": "qr generation failed: disk write failed"}
