from pathlib import Path

import pytest

from littercoast.config import AppEnvironment, DEFAULT_CLASSES, DetectionConfig, InferenceConfig, TrainingConfig


@pytest.mark.unit
def test_detection_config_builds_class_to_id_mapping() -> None:
    config = DetectionConfig()

    assert config.classes == DEFAULT_CLASSES
    assert config.class_to_id["pet_bottle"] == 0
    assert config.class_to_id["fragment"] == len(DEFAULT_CLASSES) - 1


@pytest.mark.unit
def test_training_config_computes_train_and_validation_roots() -> None:
    config = TrainingConfig()

    assert config.train_root.parts[-2:] == ("garbage_classification", "train")
    assert config.validation_root.parts[-2:] == ("garbage_classification", "val")


@pytest.mark.unit
def test_training_config_uses_local_environment_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LITTERCOAST_ENVIRONMENT", AppEnvironment.LOCAL.value)
    monkeypatch.delenv("LITTERCOAST_DATASET_ARCHIVE_PATH", raising=False)
    monkeypatch.delenv("LITTERCOAST_EXTRACTED_DIR", raising=False)
    monkeypatch.delenv("LITTERCOAST_DATASET_ROOT", raising=False)
    monkeypatch.delenv("LITTERCOAST_DATASET_YAML_PATH", raising=False)

    config = TrainingConfig()

    assert config.dataset_archive_path == DEFAULT_LOCAL_TRAINING_ARCHIVE_PATH
    assert config.dataset_root == DEFAULT_LOCAL_DATASET_ROOT


@pytest.mark.unit
def test_inference_config_for_environment_uses_local_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LITTERCOAST_ENVIRONMENT", AppEnvironment.COLAB.value)
    monkeypatch.delenv("LITTERCOAST_MODEL_PATH", raising=False)
    monkeypatch.delenv("LITTERCOAST_IMAGE_DIRECTORY", raising=False)
    monkeypatch.delenv("LITTERCOAST_PREDICTIONS_PATH", raising=False)

    config = InferenceConfig.for_environment(AppEnvironment.LOCAL)

    assert config.model_path == Path("models/yolov8_model.pt")
    assert config.image_directory == Path("data/images")
    assert config.predictions_path == Path("outputs/yolo_predictions.json")


DEFAULT_LOCAL_TRAINING_ARCHIVE_PATH = Path("data/dataset.tar.gz")
DEFAULT_LOCAL_DATASET_ROOT = Path("data/garbage_classification")
