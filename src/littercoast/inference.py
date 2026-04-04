import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from .config import InferenceConfig


class PredictionRepository:
    """Persists inference results in a JSON file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def load(self) -> list[dict]:
        if not self.output_path.exists():
            return []
        return json.loads(self.output_path.read_text(encoding="utf-8"))

    def append(self, image_name: str, predictions: list[dict]) -> None:
        saved_predictions = self.load()
        saved_predictions.append({"image": image_name, "predictions": predictions})
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(saved_predictions, indent=4),
            encoding="utf-8",
        )
        print(f"Saved prediction for {image_name}")


class ObjectDetector:
    """Runs YOLO object detection over image files."""

    def __init__(self, model_path: Path) -> None:
        self.model = YOLO(model_path)

    def detect(self, image_path: Path) -> list[dict]:
        image = Image.open(image_path)
        results = self.model(image)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                detected_objects.append(
                    {
                        "class": int(box.cls[0]),
                        "coordinates": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                    }
                )

        return detected_objects


class InferencePipeline:
    """Coordinates batch image inference and persistence."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.repository = PredictionRepository(self.config.predictions_path)
        self.detector = ObjectDetector(self.config.model_path)

    def run(self) -> list[dict]:
        for image_path in sorted(self.config.image_directory.iterdir()):
            if image_path.suffix.lower() not in self.config.allowed_extensions:
                continue
            predictions = self.detector.detect(image_path)
            self.repository.append(image_path.name, predictions)

        saved_predictions = self.repository.load()
        print(saved_predictions)
        return saved_predictions
