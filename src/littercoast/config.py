from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .env import get_env_float, get_env_int, get_env_path, get_env_str, load_env_file


if sys.version_info >= (3, 10):
    from enum import StrEnum
else:  # pragma: no cover - Python 3.9 compatibility
    class StrEnum(str, Enum):
        """Compatibility shim for Python 3.9."""


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


class AppEnvironment(StrEnum):
    COLAB = "colab"
    LOCAL = "local"
    CUSTOM = "custom"


def get_default_environment() -> AppEnvironment:
    return AppEnvironment(get_env_str("LITTERCOAST_ENVIRONMENT", AppEnvironment.LOCAL))


def _resolve_environment(environment: AppEnvironment | None = None) -> AppEnvironment:
    return environment or get_default_environment()


def _resolve_path(
    name: str,
    fallback_by_environment: dict[AppEnvironment, str],
    environment: AppEnvironment | None = None,
) -> Path:
    environment = _resolve_environment(environment)
    fallback = fallback_by_environment.get(environment, fallback_by_environment[AppEnvironment.CUSTOM])
    return get_env_path(name, fallback)


def _resolve_str(
    name: str,
    fallback_by_environment: dict[AppEnvironment, str],
    environment: AppEnvironment | None = None,
) -> str:
    environment = _resolve_environment(environment)
    fallback = fallback_by_environment.get(environment, fallback_by_environment[AppEnvironment.CUSTOM])
    return get_env_str(name, fallback)


TRAINING_ARCHIVE_DEFAULTS = {
    AppEnvironment.COLAB: "/content/drive/MyDrive/dataset.tar.gz",
    AppEnvironment.LOCAL: "./data/dataset.tar.gz",
    AppEnvironment.CUSTOM: "./data/dataset.tar.gz",
}
TRAINING_EXTRACTED_DEFAULTS = {
    AppEnvironment.COLAB: "/content/datasets/beach_plastic_litter_dataset_v2",
    AppEnvironment.LOCAL: "./data/extracted/beach_plastic_litter_dataset_v2",
    AppEnvironment.CUSTOM: "./data/extracted/beach_plastic_litter_dataset_v2",
}
TRAINING_ROOT_DEFAULTS = {
    AppEnvironment.COLAB: "/content/datasets/garbage_classification",
    AppEnvironment.LOCAL: "./data/garbage_classification",
    AppEnvironment.CUSTOM: "./data/garbage_classification",
}
TRAINING_YAML_DEFAULTS = {
    AppEnvironment.COLAB: "/content/datasets/waste.yaml",
    AppEnvironment.LOCAL: "./data/waste.yaml",
    AppEnvironment.CUSTOM: "./data/waste.yaml",
}
INFERENCE_PREDICTIONS_DEFAULTS = {
    AppEnvironment.COLAB: "/content/drive/My Drive/yolo_predictions.json",
    AppEnvironment.LOCAL: "./outputs/yolo_predictions.json",
    AppEnvironment.CUSTOM: "./outputs/yolo_predictions.json",
}
INFERENCE_MODEL_DEFAULTS = {
    AppEnvironment.COLAB: "/content/drive/My Drive/yolov8_model.pt",
    AppEnvironment.LOCAL: "./models/yolov8_model.pt",
    AppEnvironment.CUSTOM: "./models/yolov8_model.pt",
}
INFERENCE_IMAGE_DEFAULTS = {
    AppEnvironment.COLAB: "/content/drive/My Drive/images",
    AppEnvironment.LOCAL: "./data/images",
    AppEnvironment.CUSTOM: "./data/images",
}
QR_OUTPUT_DEFAULTS = {
    AppEnvironment.COLAB: "qrcode_link.png",
    AppEnvironment.LOCAL: "./outputs/qrcode_link.png",
    AppEnvironment.CUSTOM: "./outputs/qrcode_link.png",
}


@dataclass(slots=True)
class DetectionConfig:
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))

    @property
    def class_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}


@dataclass(slots=True)
class TrainingConfig:
    dataset_archive_path: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_DATASET_ARCHIVE_PATH", TRAINING_ARCHIVE_DEFAULTS)
    )
    extracted_dir: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_EXTRACTED_DIR", TRAINING_EXTRACTED_DEFAULTS)
    )
    dataset_root: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_DATASET_ROOT", TRAINING_ROOT_DEFAULTS)
    )
    dataset_yaml_path: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_DATASET_YAML_PATH", TRAINING_YAML_DEFAULTS)
    )
    model_name: str = field(
        default_factory=lambda: _resolve_str(
            "LITTERCOAST_MODEL_NAME",
            {
                AppEnvironment.COLAB: "yolov8n.pt",
                AppEnvironment.LOCAL: "yolov8n.pt",
                AppEnvironment.CUSTOM: "yolov8n.pt",
            },
        )
    )
    epochs: int = field(default_factory=lambda: get_env_int("LITTERCOAST_EPOCHS", 50))
    image_size: int = field(default_factory=lambda: get_env_int("LITTERCOAST_IMAGE_SIZE", 640))
    batch_size: int = field(default_factory=lambda: get_env_int("LITTERCOAST_BATCH_SIZE", 16))
    run_name: str = field(default_factory=lambda: get_env_str("LITTERCOAST_RUN_NAME", "garbage_detection_yolov8"))
    train_split_ratio: float = field(default_factory=lambda: get_env_float("LITTERCOAST_TRAIN_SPLIT_RATIO", 0.8))

    @classmethod
    def for_environment(cls, environment: AppEnvironment | None = None) -> "TrainingConfig":
        return cls(
            dataset_archive_path=_resolve_path("LITTERCOAST_DATASET_ARCHIVE_PATH", TRAINING_ARCHIVE_DEFAULTS, environment),
            extracted_dir=_resolve_path("LITTERCOAST_EXTRACTED_DIR", TRAINING_EXTRACTED_DEFAULTS, environment),
            dataset_root=_resolve_path("LITTERCOAST_DATASET_ROOT", TRAINING_ROOT_DEFAULTS, environment),
            dataset_yaml_path=_resolve_path("LITTERCOAST_DATASET_YAML_PATH", TRAINING_YAML_DEFAULTS, environment),
            model_name=_resolve_str(
                "LITTERCOAST_MODEL_NAME",
                {
                    AppEnvironment.COLAB: "yolov8n.pt",
                    AppEnvironment.LOCAL: "yolov8n.pt",
                    AppEnvironment.CUSTOM: "yolov8n.pt",
                },
                environment,
            ),
            epochs=get_env_int("LITTERCOAST_EPOCHS", 50),
            image_size=get_env_int("LITTERCOAST_IMAGE_SIZE", 640),
            batch_size=get_env_int("LITTERCOAST_BATCH_SIZE", 16),
            run_name=get_env_str("LITTERCOAST_RUN_NAME", "garbage_detection_yolov8"),
            train_split_ratio=get_env_float("LITTERCOAST_TRAIN_SPLIT_RATIO", 0.8),
        )

    @property
    def train_root(self) -> Path:
        return self.dataset_root / "train"

    @property
    def validation_root(self) -> Path:
        return self.dataset_root / "val"


@dataclass(slots=True)
class InferenceConfig:
    predictions_path: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_PREDICTIONS_PATH", INFERENCE_PREDICTIONS_DEFAULTS)
    )
    model_path: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_MODEL_PATH", INFERENCE_MODEL_DEFAULTS)
    )
    image_directory: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_IMAGE_DIRECTORY", INFERENCE_IMAGE_DEFAULTS)
    )
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    @classmethod
    def for_environment(cls, environment: AppEnvironment | None = None) -> "InferenceConfig":
        return cls(
            predictions_path=_resolve_path("LITTERCOAST_PREDICTIONS_PATH", INFERENCE_PREDICTIONS_DEFAULTS, environment),
            model_path=_resolve_path("LITTERCOAST_MODEL_PATH", INFERENCE_MODEL_DEFAULTS, environment),
            image_directory=_resolve_path("LITTERCOAST_IMAGE_DIRECTORY", INFERENCE_IMAGE_DEFAULTS, environment),
        )


@dataclass(slots=True)
class QRCodeConfig:
    link: str = field(default_factory=lambda: get_env_str("LITTERCOAST_QR_LINK", "https://forms.gle/PLDgtbQkSxKboPfv7"))
    output_path: Path = field(
        default_factory=lambda: _resolve_path("LITTERCOAST_QR_OUTPUT_PATH", QR_OUTPUT_DEFAULTS)
    )
    version: int = field(default_factory=lambda: get_env_int("LITTERCOAST_QR_VERSION", 1))
    box_size: int = field(default_factory=lambda: get_env_int("LITTERCOAST_QR_BOX_SIZE", 10))
    border: int = field(default_factory=lambda: get_env_int("LITTERCOAST_QR_BORDER", 4))

    @classmethod
    def for_environment(cls, environment: AppEnvironment | None = None) -> "QRCodeConfig":
        return cls(
            link=get_env_str("LITTERCOAST_QR_LINK", "https://forms.gle/PLDgtbQkSxKboPfv7"),
            output_path=_resolve_path("LITTERCOAST_QR_OUTPUT_PATH", QR_OUTPUT_DEFAULTS, environment),
            version=get_env_int("LITTERCOAST_QR_VERSION", 1),
            box_size=get_env_int("LITTERCOAST_QR_BOX_SIZE", 10),
            border=get_env_int("LITTERCOAST_QR_BORDER", 4),
        )
