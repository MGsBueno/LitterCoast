from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .env import get_env_float, get_env_int, get_env_path, get_env_str, load_env_file


load_env_file()


DEFAULT_CLASSES = [
    "pet_bottle",
    "other_bottle",
    "plastic_bag",
    "box_shaped_case",
    "other_container",
    "rope",
    "other_string",
    "fishing_net",
    "buoy",
    "other_fishing_gear",
    "styrene_foam",
    "others",
    "fragment",
]


@dataclass(slots=True)
class DetectionConfig:
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))

    @property
    def class_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}


@dataclass(slots=True)
class TrainingConfig:
    dataset_archive_path: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_DATASET_ARCHIVE_PATH", "/content/drive/MyDrive/dataset.tar.gz")
    )
    extracted_dir: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_EXTRACTED_DIR", "/content/datasets/beach_plastic_litter_dataset_v2")
    )
    dataset_root: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_DATASET_ROOT", "/content/datasets/garbage_classification")
    )
    dataset_yaml_path: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_DATASET_YAML_PATH", "/content/datasets/waste.yaml")
    )
    model_name: str = field(default_factory=lambda: get_env_str("LITTERCOAST_MODEL_NAME", "yolov8n.pt"))
    epochs: int = field(default_factory=lambda: get_env_int("LITTERCOAST_EPOCHS", 50))
    image_size: int = field(default_factory=lambda: get_env_int("LITTERCOAST_IMAGE_SIZE", 640))
    batch_size: int = field(default_factory=lambda: get_env_int("LITTERCOAST_BATCH_SIZE", 16))
    run_name: str = field(default_factory=lambda: get_env_str("LITTERCOAST_RUN_NAME", "garbage_detection_yolov8"))
    train_split_ratio: float = field(default_factory=lambda: get_env_float("LITTERCOAST_TRAIN_SPLIT_RATIO", 0.8))

    @property
    def train_root(self) -> Path:
        return self.dataset_root / "train"

    @property
    def validation_root(self) -> Path:
        return self.dataset_root / "val"


@dataclass(slots=True)
class InferenceConfig:
    predictions_path: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_PREDICTIONS_PATH", "/content/drive/My Drive/yolo_predictions.json")
    )
    model_path: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_MODEL_PATH", "/content/drive/My Drive/yolov8_model.pt")
    )
    image_directory: Path = field(
        default_factory=lambda: get_env_path("LITTERCOAST_IMAGE_DIRECTORY", "/content/drive/My Drive/images")
    )
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass(slots=True)
class QRCodeConfig:
    link: str = field(default_factory=lambda: get_env_str("LITTERCOAST_QR_LINK", "https://forms.gle/PLDgtbQkSxKboPfv7"))
    output_path: Path = field(default_factory=lambda: get_env_path("LITTERCOAST_QR_OUTPUT_PATH", "qrcode_link.png"))
    version: int = field(default_factory=lambda: get_env_int("LITTERCOAST_QR_VERSION", 1))
    box_size: int = field(default_factory=lambda: get_env_int("LITTERCOAST_QR_BOX_SIZE", 10))
    border: int = field(default_factory=lambda: get_env_int("LITTERCOAST_QR_BORDER", 4))
