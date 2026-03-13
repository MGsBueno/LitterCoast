"""
Dependency installation for Google Colab.
"""

#!pip install opencv-python pycocotools tqdm ultralytics

import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import zipfile

import cv2
from pycocotools import mask
from ultralytics import YOLO

"""
Dependency installation for local execution.
"""

subprocess.run(["ls", "-l"])
subprocess.run(["pip", "install", "opencv-python", "pycocotools", "tqdm", "ultralytics"])

# Mount Google Drive to access the dataset.
# drive.mount("/content/drive")

# Path to the dataset archive.
dataset_archive_path = "/content/drive/MyDrive/dataset.tar.gz"
# dataset_archive_path = "/content/drive/MyDrive/dataset.zip"


def extract_archive(archive_path, output_dir):
    """Extract a .tar.gz or .zip archive into the target directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(path=output_dir)
            print(f"Extracted .tar archive to: {output_dir}")
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
            print(f"Extracted .zip archive to: {output_dir}")
    else:
        raise ValueError("Unsupported file format. Use .zip or .tar.gz")


archive_file = "/content/drive/MyDrive/dataset.tar.gz"
extracted_dir = "/content/datasets/beach_plastic_litter_dataset_v2"

extract_archive(archive_file, extracted_dir)
extract_archive(dataset_archive_path, extracted_dir)

# Create training and validation directories if they do not exist.
dataset_root = "/content/datasets/garbage_classification"
os.makedirs(os.path.join(dataset_root, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_root, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(dataset_root, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_root, "val", "labels"), exist_ok=True)

# Create the YAML configuration file used by YOLOv8 training.
dataset_yaml_path = "/content/datasets/waste.yaml"
with open(dataset_yaml_path, "w") as yaml_file:
    yaml_file.write(
        """
path: /content/datasets/waste
train: /content/datasets/train/images
val: /content/datasets/val/images

nc: 13
names: ['pet_bottle', 'other_bottle', 'plastic_bag', 'box_shaped_case', 'other_container', 'rope', 'other_string', 'fishing_net', 'buoy', 'other_fishing_gear', 'styrene_foam', 'others', 'fragment']
"""
    )
    print("YAML configuration file created successfully.")


def extract_class_from_filename(filename):
    """Extract the class prefix from a file name."""
    match = re.match(r"([a-zA-Z]+)", filename)
    return match.group(1).lower() if match else None


def process_annotations(json_file, images_dir, labels_dir, classes):
    with open(json_file, "r") as file:
        annotations = json.load(file)

    class_to_id = {label: index for index, label in enumerate(classes)}

    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        category_name = [c["name"] for c in annotations["categories"] if c["id"] == category_id][0]
        class_id = class_to_id[category_name]

        image_info = next(image for image in annotations["images"] if image["id"] == image_id)
        image_name = image_info["file_name"]
        image_path = os.path.join(images_dir, image_name)

        run_length_encoding = annotation["segmentation"]
        mask_array = mask.decode(run_length_encoding)

        label_file = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(label_file, "w") as output_file:
            polygons = mask.encode(mask_array)
            output_file.write(f"{class_id} {polygons}\n")


annotations_file = os.path.join(extracted_dir, "annotations.json")

classes = [
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

process_annotations(
    annotations_file,
    "/content/datasets/train/images",
    "/content/datasets/train/labels",
    classes,
)

print("Annotations processed and labels generated successfully.")

model = YOLO("yolov8n.pt")

model.train(
    data=dataset_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="garbage_segmentation_yolov8",
    task="segment",
)

image_path = "/content/datasets/train/images/example_image.jpg"
results = model(image_path)
