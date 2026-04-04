from pathlib import Path

import pytest

from littercoast.config import InferenceConfig
from littercoast.inference import InferencePipeline, PredictionRepository


@pytest.mark.unit
def test_prediction_repository_appends_and_loads_predictions(workspace_dir: Path) -> None:
    repository = PredictionRepository(workspace_dir / "predictions.json")

    repository.append("image-1.jpg", [{"class": 0, "confidence": 0.9}])
    repository.append("image-2.jpg", [{"class": 1, "confidence": 0.8}])

    assert repository.load() == [
        {"image": "image-1.jpg", "predictions": [{"class": 0, "confidence": 0.9}]},
        {"image": "image-2.jpg", "predictions": [{"class": 1, "confidence": 0.8}]},
    ]


class _FakeDetector:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def detect(self, image_path: Path) -> list[dict]:
        self.calls.append(image_path)
        return [{"class": 0, "coordinates": [0, 0, 10, 10], "confidence": 0.95}]


@pytest.mark.unit
def test_inference_pipeline_processes_supported_images_only(monkeypatch, workspace_dir: Path) -> None:
    image_directory = workspace_dir / "images"
    image_directory.mkdir()
    (image_directory / "sample-1.jpg").write_bytes(b"jpg")
    (image_directory / "sample-2.png").write_bytes(b"png")
    (image_directory / "ignore.txt").write_text("skip", encoding="utf-8")

    config = InferenceConfig(
        model_path=workspace_dir / "model.pt",
        image_directory=image_directory,
        predictions_path=workspace_dir / "predictions.json",
    )

    fake_detector = _FakeDetector()
    monkeypatch.setattr("littercoast.inference.ObjectDetector", lambda model_path: fake_detector)

    pipeline = InferencePipeline(config)
    saved_predictions = pipeline.run()

    assert [path.name for path in fake_detector.calls] == ["sample-1.jpg", "sample-2.png"]
    assert len(saved_predictions) == 2
    assert saved_predictions[0]["image"] == "sample-1.jpg"
    assert saved_predictions[1]["image"] == "sample-2.png"
