import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from ultralytics import YOLO

from .config import DetectionConfig, TrainingConfig


class ArchiveExtractor:
    """Handles extraction of supported dataset archives."""

    def extract(self, archive_path: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as archive:
                self._validate_tar_members(archive, output_dir)
                archive.extractall(path=output_dir)
            print(f"Extracted tar archive to: {output_dir}")
            return

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                self._validate_zip_members(archive, output_dir)
                archive.extractall(output_dir)
            print(f"Extracted zip archive to: {output_dir}")
            return

        raise ValueError(f"Unsupported archive format: {archive_path}")

    def _validate_tar_members(self, archive: tarfile.TarFile, output_dir: Path) -> None:
        for member in archive.getmembers():
            self._validate_extraction_path(output_dir, member.name)

    def _validate_zip_members(self, archive: zipfile.ZipFile, output_dir: Path) -> None:
        for member in archive.infolist():
            self._validate_extraction_path(output_dir, member.filename)

    def _validate_extraction_path(self, output_dir: Path, member_name: str) -> None:
        destination = (output_dir / member_name).resolve()
        output_root = output_dir.resolve()
        try:
            is_within_output_dir = Path(destination).is_relative_to(output_root)
        except AttributeError:
            is_within_output_dir = str(destination).startswith(str(output_root))

        if not is_within_output_dir:
            raise ValueError(f"Unsafe archive member path detected: {member_name}")


class DatasetPreparer:
    """Builds YOLO training folders and placeholder labels from class folders."""

    def __init__(self, detection_config: DetectionConfig, training_config: TrainingConfig) -> None:
        self.detection_config = detection_config
        self.training_config = training_config

    def prepare(self) -> None:
        train_images_dir = self.training_config.train_root / "images"
        train_labels_dir = self.training_config.train_root / "labels"
        validation_images_dir = self.training_config.validation_root / "images"
        validation_labels_dir = self.training_config.validation_root / "labels"

        for directory in (
            train_images_dir,
            train_labels_dir,
            validation_images_dir,
            validation_labels_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        for category_name in self.detection_config.classes:
            source_dir = self.training_config.extracted_dir / category_name
            if not source_dir.exists():
                print(f"Folder not found: {source_dir}. Skipping.")
                continue

            image_files = [
                path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            random.shuffle(image_files)

            split_index = int(len(image_files) * self.training_config.train_split_ratio)
            train_images = image_files[:split_index]
            validation_images = image_files[split_index:]

            self._copy_images_and_labels(train_images, train_images_dir, train_labels_dir, category_name)
            self._copy_images_and_labels(
                validation_images,
                validation_images_dir,
                validation_labels_dir,
                category_name,
            )

        print("Image and label organization completed.")

    def write_dataset_yaml(self) -> None:
        dataset_yaml = "\n".join(
            [
                f"path: {self.training_config.dataset_root}",
                f"train: {self.training_config.train_root / 'images'}",
                f"val: {self.training_config.validation_root / 'images'}",
                "",
                f"nc: {len(self.detection_config.classes)}",
                f"names: {self.detection_config.classes}",
                "",
            ]
        )
        self.training_config.dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        self.training_config.dataset_yaml_path.write_text(dataset_yaml, encoding="utf-8")
        print(f"Dataset YAML created at: {self.training_config.dataset_yaml_path}")

    def _copy_images_and_labels(
        self,
        image_paths: list[Path],
        destination_images_dir: Path,
        destination_labels_dir: Path,
        category_name: str,
    ) -> None:
        class_id = self.detection_config.class_to_id[category_name]

        for image_path in image_paths:
            destination_image_path = destination_images_dir / image_path.name
            shutil.copy(image_path, destination_image_path)

            destination_label_path = destination_labels_dir / f"{image_path.stem}.txt"
            destination_label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


class ModelTrainer:
    """Coordinates dataset extraction, preparation, and YOLO training."""

    def __init__(
        self,
        detection_config: DetectionConfig | None = None,
        training_config: TrainingConfig | None = None,
        extractor: ArchiveExtractor | None = None,
    ) -> None:
        self.detection_config = detection_config or DetectionConfig()
        self.training_config = training_config or TrainingConfig()
        self.extractor = extractor or ArchiveExtractor()
        self.dataset_preparer = DatasetPreparer(self.detection_config, self.training_config)

    def run(self) -> None:
        self.training_config.dataset_root.mkdir(parents=True, exist_ok=True)
        self.extractor.extract(self.training_config.dataset_archive_path, self.training_config.extracted_dir)
        self.dataset_preparer.write_dataset_yaml()
        self.dataset_preparer.prepare()

        model = YOLO(self.training_config.model_name)
        model.train(
            data=str(self.training_config.dataset_yaml_path),
            epochs=self.training_config.epochs,
            imgsz=self.training_config.image_size,
            batch=self.training_config.batch_size,
            name=self.training_config.run_name,
            task="detect",
        )
