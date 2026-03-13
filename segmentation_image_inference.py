import json
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO

# Path to the JSON file used to store predictions.
json_file_path = "/content/drive/My Drive/yolov8_segmentation_predictions.json"


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


model = YOLO("/content/drive/My Drive/yolov8_model.pt")


def segment_objects(image, model):
    results = model(image)
    segmented_objects = []

    for result in results:
        for prediction in result.boxes.data:
            detected_class = int(prediction[5].item())
            confidence = prediction[4].item()
            coordinates = prediction[:4].tolist()

            if "mask" in result.masks:
                segmentation_mask = result.masks.data.cpu().numpy().tolist()
            else:
                segmentation_mask = None

            segmented_objects.append(
                {
                    "class": detected_class,
                    "coordinates": coordinates,
                    "confidence": confidence,
                    "mask": segmentation_mask,
                }
            )

    return segmented_objects


image_directory = "/content/drive/My Drive/images/"

for image_name in os.listdir(image_directory):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image = Image.open(os.path.join(image_directory, image_name))
        predictions = segment_objects(image, model)
        save_prediction(image_name, predictions)

saved_predictions = load_predictions()
print(saved_predictions)
