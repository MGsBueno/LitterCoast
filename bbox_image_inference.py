import json
import os

from PIL import Image
from ultralytics import YOLO

# Path to the JSON file used to store predictions.
json_file_path = "/content/drive/My Drive/yolo_predictions.json"
model_path = "/content/drive/My Drive/yolov8_model.pt"


def save_prediction(image_name, predictions):
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            saved_predictions = json.load(file)
    else:
        saved_predictions = []

    saved_predictions.append({"image": image_name, "predictions": predictions})

    with open(json_file_path, "w") as file:
        json.dump(saved_predictions, file, indent=4)

    print(f"Saved prediction for {image_name}")


def load_predictions():
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            saved_predictions = json.load(file)
        return saved_predictions
    return []


def detect_objects(image, model):
    results = model(image)
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


# Path to the image directory in Google Drive.
image_directory = "/content/drive/My Drive/images/"
model = YOLO(model_path)

for image_name in os.listdir(image_directory):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image = Image.open(os.path.join(image_directory, image_name))
        predictions = detect_objects(image, model)
        save_prediction(image_name, predictions)

saved_predictions = load_predictions()
print(saved_predictions)
