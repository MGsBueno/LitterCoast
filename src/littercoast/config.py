from dataclasses import dataclass, field
from pathlib import Path


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
    dataset_archive_path: Path = Path("/content/drive/MyDrive/dataset.tar.gz")
    extracted_dir: Path = Path("/content/datasets/beach_plastic_litter_dataset_v2")
    dataset_root: Path = Path("/content/datasets/garbage_classification")
    dataset_yaml_path: Path = Path("/content/datasets/waste.yaml")
    model_name: str = "yolov8n.pt"
    epochs: int = 50
    image_size: int = 640
    batch_size: int = 16
    run_name: str = "garbage_detection_yolov8"
    train_split_ratio: float = 0.8

    @property
    def train_root(self) -> Path:
        return self.dataset_root / "train"

    @property
    def validation_root(self) -> Path:
        return self.dataset_root / "val"


@dataclass(slots=True)
class InferenceConfig:
    predictions_path: Path = Path("/content/drive/My Drive/yolo_predictions.json")
    model_path: Path = Path("/content/drive/My Drive/yolov8_model.pt")
    image_directory: Path = Path("/content/drive/My Drive/images")
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass(slots=True)
class QRCodeConfig:
    link: str = "https://forms.gle/PLDgtbQkSxKboPfv7"
    output_path: Path = Path("qrcode_link.png")
    version: int = 1
    box_size: int = 10
    border: int = 4
