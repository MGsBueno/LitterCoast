import os
import subprocess
# Verificação e instalação das dependências (caso necessário)
subprocess.run(["pip", "install", "opencv-python", "pycocotools", "tqdm", "ultralytics"])


import zipfile
import shutil
import random
import re
import json
import tarfile
from ultralytics import YOLO
from pycocotools import mask


# Montar o Google Drive para acesso
#drive.mount('/content/drive')  # Mapeia seu Google Drive para /content/drive

# Caminho do arquivo .tar.gz ou zip que contém o dataset
file = '/content/drive/MyDrive/dataset.tar.gz'
#file = '/content/drive/MyDrive/dataset.zip'


# Criação da pasta datasets
os.makedirs("/content/datasets", exist_ok=True)

# Diretório de destino onde os arquivos serão extraídos
extracted_dir = '/content/datasets/beach_plastic_litter_dataset'


# Função para extrair o arquivo .tar.gz
def extract_archive(file_path, extracted_dir):
    """Extrai arquivos .tar.gz ou .zip para o diretório especificado."""
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)

    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, "r:*") as tar:
            tar.extractall(path=extracted_dir)
            print(f"Arquivo .tar extraído para: {extracted_dir}")
    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
            print(f"Arquivo .zip extraído para: {extracted_dir}")
    else:
        raise ValueError("Formato de arquivo não suportado. Use .zip ou .tar.gz")




# Diretório de destino onde os arquivos serão extraídos
extracted_dir = '/content/datasets/beach_plastic_litter_dataset_v2'

# Extrair o dataset
extract_archive(archive_file, extracted_dir)
# Extrair o dataset
extract_archive(file, extracted_dir)

"""
Criar diretórios de treino e validação (se não existirem)
"""
DATASET_PATH = "/content/datasets/garbage_classification"
os.makedirs(os.path.join(DATASET_PATH, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "val", "labels"), exist_ok=True)

"""
Criar arquivo YAML de configuração
Este arquivo YAML será usado pelo YOLOv8 para configurar o treinamento e as validações do modelo
"""
dataset_yaml = "/content/datasets/lixo.yaml"
with open(dataset_yaml, "w") as f:
    f.write("""
path: /content/datasets/lixo
train: /content/datasets/train/images
val: /content/datasets/val/images

nc: 13
names: ['pet_bottle', 'other_bottle', 'plastic_bag', 'box_shaped_case', 'other_container', 'rope', 'other_string', 'fishing_net', 'buoy', 'other_fishing_gear', 'styrene_foam', 'others', 'fragment']
""")
    print("Arquivo YAML criado com sucesso!")

"""
Função para extrair a classe de um arquivo de imagem com base no nome do arquivo
Essa função ajuda a mapear os arquivos de imagem para suas respectivas classes (rótulos de lixo)
"""
def extract_class_from_filename(filename):
    match = re.match(r"([a-zA-Z]+)", filename)  # Pega a primeira palavra (antes dos números)
    return match.group(1).lower() if match else None

"""
Função para processar as anotações COCO e gerar os arquivos de segmentação para YOLO
A função converte as anotações COCO para o formato de segmentação que o YOLO entende
"""
# Separar imagens por classe (dataset com 6 classes)
orig_path = "/content/datasets/6classes v2"
train_img_dir = "/content/datasets/garbage-classification-6-classes-775class/train/images"
val_img_dir = "/content/datasets/garbage-classification-6-classes-775class/val/images"
train_lbl_dir = "/content/datasets/garbage-classification-6-classes-775class/train/labels"
val_lbl_dir = "/content/datasets/garbage-classification-6-classes-775class/val/labels"

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

classes = ['pet_bottle', 'other_bottle', 'plastic_bag',  'box_shaped_case', 'other_container', 'rope', 'other_string', 'fishing_net', 'buoy', 'other_fishing_gear', 'styrene_foam', 'others', 'fragment']


class_to_id = {cls: i for i, cls in enumerate(classes)}

for category in classes:
    category_path = os.path.join(orig_path, category)

    if not os.path.exists(category_path):
        print(f"Pasta {category_path} não encontrada! Pulando...")
        continue

    images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_img_dir, img))
        label_path = os.path.join(train_lbl_dir, os.path.splitext(img)[0] + ".txt")
        with open(label_path, "w") as label_f:
            label_f.write(f"{class_to_id[category]} 0.5 0.5 1.0 1.0\n")  # Dummy bbox

    for img in val_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(val_img_dir, img))
        label_path = os.path.join(val_lbl_dir, os.path.splitext(img)[0] + ".txt")
        with open(label_path, "w") as label_f:
            label_f.write(f"{class_to_id[category]} 0.5 0.5 1.0 1.0\n")  # Dummy bbox

print("Organização de imagens e rótulos do segundo dataset concluída.")

# Carrega o modelo de detecção (não segmentação)
model = YOLO("yolov8n.pt")  # ou yolov8s.pt, yolov8m.pt, yolov8l.pt

# Inicia o treinamento
model.train(
    data="/content/datasets/lixo.yaml",  # Caminho para o arquivo YAML
    epochs=50,
    imgsz=640,
    batch=16,
    name="garbage_detection_yolov8",
    task="detect"  # <-- importante: tarefa de DETECÇÃO
)
