from pathlib import Path
import tarfile
import zipfile

import pytest

from littercoast.config import DetectionConfig, TrainingConfig
from littercoast.training import ArchiveExtractor, DatasetPreparer, ModelTrainer


def _create_fake_image(path: Path) -> None:
    path.write_bytes(b"fake-image-content")


@pytest.mark.unit
def test_dataset_preparer_creates_split_structure_and_labels(workspace_dir: Path) -> None:
    detection_config = DetectionConfig(classes=["pet_bottle"])
    extracted_dir = workspace_dir / "source"
    class_dir = extracted_dir / "pet_bottle"
    class_dir.mkdir(parents=True)

    for image_name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        _create_fake_image(class_dir / image_name)

    training_config = TrainingConfig(
        extracted_dir=extracted_dir,
        dataset_root=workspace_dir / "dataset",
        dataset_yaml_path=workspace_dir / "dataset" / "waste.yaml",
        train_split_ratio=0.5,
    )

    preparer = DatasetPreparer(detection_config, training_config)
    preparer.prepare()

    train_images = sorted((training_config.train_root / "images").iterdir())
    validation_images = sorted((training_config.validation_root / "images").iterdir())
    train_labels = sorted((training_config.train_root / "labels").iterdir())
    validation_labels = sorted((training_config.validation_root / "labels").iterdir())

    assert len(train_images) == 2
    assert len(validation_images) == 2
    assert len(train_labels) == 2
    assert len(validation_labels) == 2
    assert all(
        label.read_text(encoding="utf-8") == "0 0.5 0.5 1.0 1.0\n"
        for label in train_labels + validation_labels
    )


@pytest.mark.unit
def test_dataset_preparer_writes_yaml_with_expected_content(workspace_dir: Path) -> None:
    detection_config = DetectionConfig(classes=["pet_bottle", "rope"])
    training_config = TrainingConfig(
        dataset_root=workspace_dir / "dataset",
        dataset_yaml_path=workspace_dir / "config" / "waste.yaml",
    )

    preparer = DatasetPreparer(detection_config, training_config)
    preparer.write_dataset_yaml()

    content = training_config.dataset_yaml_path.read_text(encoding="utf-8")

    assert f"path: {training_config.dataset_root}" in content
    assert f"train: {training_config.train_root / 'images'}" in content
    assert f"val: {training_config.validation_root / 'images'}" in content
    assert "nc: 2" in content
    assert "names: ['pet_bottle', 'rope']" in content


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def extract(self, archive_path: Path, output_dir: Path) -> None:
        self.calls.append((archive_path, output_dir))


class _FakeTrainerModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.train_calls: list[dict] = []

    def train(self, **kwargs) -> None:
        self.train_calls.append(kwargs)


@pytest.mark.unit
def test_model_trainer_orchestrates_extraction_preparation_and_training(monkeypatch, workspace_dir: Path) -> None:
    detection_config = DetectionConfig(classes=["pet_bottle"])
    training_config = TrainingConfig(
        dataset_archive_path=workspace_dir / "dataset.zip",
        extracted_dir=workspace_dir / "extracted",
        dataset_root=workspace_dir / "dataset",
        dataset_yaml_path=workspace_dir / "dataset" / "waste.yaml",
        model_name="mock-model.pt",
        epochs=2,
        image_size=320,
        batch_size=4,
        run_name="unit-test-run",
    )
    training_config.dataset_archive_path.write_text("placeholder", encoding="utf-8")

    fake_extractor = _FakeExtractor()
    fake_model = _FakeTrainerModel(training_config.model_name)

    monkeypatch.setattr("littercoast.training.YOLO", lambda model_name: fake_model)

    trainer = ModelTrainer(
        detection_config=detection_config,
        training_config=training_config,
        extractor=fake_extractor,
    )

    prepare_calls: list[str] = []
    yaml_calls: list[str] = []
    monkeypatch.setattr(trainer.dataset_preparer, "prepare", lambda: prepare_calls.append("prepare"))
    monkeypatch.setattr(trainer.dataset_preparer, "write_dataset_yaml", lambda: yaml_calls.append("yaml"))

    trainer.run()

    assert fake_extractor.calls == [(training_config.dataset_archive_path, training_config.extracted_dir)]
    assert prepare_calls == ["prepare"]
    assert yaml_calls == ["yaml"]
    assert fake_model.train_calls == [
        {
            "data": str(training_config.dataset_yaml_path),
            "epochs": 2,
            "imgsz": 320,
            "batch": 4,
            "name": "unit-test-run",
            "task": "detect",
        }
    ]


@pytest.mark.unit
def test_archive_extractor_rejects_unsafe_tar_members(workspace_dir: Path) -> None:
    archive_path = workspace_dir / "unsafe.tar"
    extracted_dir = workspace_dir / "extracted"

    malicious_file = workspace_dir / "payload.txt"
    malicious_file.write_text("payload", encoding="utf-8")

    with tarfile.open(archive_path, "w") as archive:
        archive.add(malicious_file, arcname="../escape.txt")

    extractor = ArchiveExtractor()

    with pytest.raises(ValueError, match="Unsafe archive member path detected"):
        extractor.extract(archive_path, extracted_dir)


@pytest.mark.unit
def test_archive_extractor_rejects_unsafe_zip_members(workspace_dir: Path) -> None:
    archive_path = workspace_dir / "unsafe.zip"
    extracted_dir = workspace_dir / "extracted"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "payload")

    extractor = ArchiveExtractor()

    with pytest.raises(ValueError, match="Unsafe archive member path detected"):
        extractor.extract(archive_path, extracted_dir)
