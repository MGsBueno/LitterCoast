import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

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
    resized_image = image.resize((416, 416))
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)

    detected_objects = []
    for prediction in predictions[0]:
        if prediction[4] > 0.5:
            detected_class = int(prediction[5])
            coordinates = prediction[:4]
            detected_objects.append(
                {
                    "class": detected_class,
                    "coordinates": coordinates,
                    "confidence": prediction[4],
                }
            )

    return detected_objects


# Path to the image directory in Google Drive.
image_directory = "/content/drive/My Drive/images/"
model = tf.keras.models.load_model(model_path)

for image_name in os.listdir(image_directory):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image = Image.open(os.path.join(image_directory, image_name))

        predictions = detect_objects(image, model)
        save_prediction(image_name, predictions)

saved_predictions = load_predictions()
print(saved_predictions)
