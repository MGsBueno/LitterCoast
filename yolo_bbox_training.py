import os
import random
import re
import shutil
import subprocess
import tarfile
import zipfile

from ultralytics import YOLO

# Check and install dependencies when needed.
subprocess.run(["pip", "install", "opencv-python", "pycocotools", "tqdm", "ultralytics"])

# Mount Google Drive to access the dataset.
# drive.mount("/content/drive")

# Path to the dataset archive.
dataset_archive_path = "/content/drive/MyDrive/dataset.tar.gz"
# dataset_archive_path = "/content/drive/MyDrive/dataset.zip"

# Create the base datasets directory.
os.makedirs("/content/datasets", exist_ok=True)


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


extracted_dir = "/content/datasets/beach_plastic_litter_dataset_v2"

extract_archive(dataset_archive_path, extracted_dir)

# Create training and validation directories if they do not exist.
dataset_root = "/content/datasets/garbage_classification"
train_root = os.path.join(dataset_root, "train")
val_root = os.path.join(dataset_root, "val")
os.makedirs(os.path.join(dataset_root, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_root, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(dataset_root, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_root, "val", "labels"), exist_ok=True)

# Create the YAML configuration file used by YOLOv8 training.
dataset_yaml_path = "/content/datasets/waste.yaml"
with open(dataset_yaml_path, "w") as yaml_file:
    yaml_file.write(
        f"""
path: {dataset_root}
train: {train_root}/images
val: {val_root}/images

nc: 13
names: ['pet_bottle', 'other_bottle', 'plastic_bag', 'box_shaped_case', 'other_container', 'rope', 'other_string', 'fishing_net', 'buoy', 'other_fishing_gear', 'styrene_foam', 'others', 'fragment']
"""
    )
    print("YAML configuration file created successfully.")


def extract_class_from_filename(filename):
    """Extract the class prefix from a file name."""
    match = re.match(r"([a-zA-Z]+)", filename)
    return match.group(1).lower() if match else None


# Split source images by class and create placeholder bbox labels.
source_dataset_dir = extracted_dir
train_image_dir = os.path.join(train_root, "images")
val_image_dir = os.path.join(val_root, "images")
train_label_dir = os.path.join(train_root, "labels")
val_label_dir = os.path.join(val_root, "labels")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

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

class_to_id = {label: index for index, label in enumerate(classes)}

for category_name in classes:
    category_path = os.path.join(source_dataset_dir, category_name)

    if not os.path.exists(category_path):
        print(f"Folder not found: {category_path}. Skipping.")
        continue

    image_files = [
        filename
        for filename in os.listdir(category_path)
        if filename.endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(image_files)

    split_index = int(len(image_files) * 0.8)
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    for image_name in train_images:
        shutil.copy(os.path.join(category_path, image_name), os.path.join(train_image_dir, image_name))
        label_path = os.path.join(train_label_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(label_path, "w") as label_file:
            label_file.write(f"{class_to_id[category_name]} 0.5 0.5 1.0 1.0\n")

    for image_name in val_images:
        shutil.copy(os.path.join(category_path, image_name), os.path.join(val_image_dir, image_name))
        label_path = os.path.join(val_label_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(label_path, "w") as label_file:
            label_file.write(f"{class_to_id[category_name]} 0.5 0.5 1.0 1.0\n")

print("Image and label organization completed.")

# Load the detection model.
model = YOLO("yolov8n.pt")

# Start bbox training.
model.train(
    data=dataset_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="garbage_detection_yolov8",
    task="detect",
)
