"""
Instalação de dependências Google collab
"""

#!pip install opencv-python pycocotools tqdm ultralytics

# Importar as bibliotecas necessárias
import os
import shutil
import random
import re
import json
import zipfile
import cv2
#from google.colab import drive
from ultralytics import YOLO
from pycocotools import mask
import tarfile


"""
Instalação de dependências para execução em máquina local
"""
import subprocess
# Exemplo: rodar 'ls -l'
subprocess.run(["ls", "-l"])
# Exemplo: rodar 'pip install opencv-python'
subprocess.run(["pip", "install", "opencv-python", "pycocotools", "tqdm", "ultralytics"])



# Montar o Google Drive para acesso
#drive.mount('/content/drive')  # Mapeia seu Google Drive para /content/drive

# Caminho do arquivo .tar.gz ou zip que contém o dataset
file = '/content/drive/MyDrive/dataset.tar.gz'
#file = '/content/drive/MyDrive/dataset.zip'

# Diretório de destino onde os arquivos serão extraídos
extracted_dir = '/content/datasets/beach_plastic_litter_dataset_v2'

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

# Caminho do arquivo de dataset (pode ser .tar.gz ou .zip)
archive_file = '/content/drive/MyDrive/dataset.tar.gz'  # ou .zip

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
def process_annotations(json_file, images_dir, labels_dir, classes):
    with open(json_file, "r") as f:
        annotations = json.load(f)

    # Criar um mapeamento das classes para IDs
    class_to_id = {cls: i for i, cls in enumerate(classes)}

    for ann in annotations["annotations"]:
        # Extrair dados de cada anotação (máscara e classe)
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = [c["name"] for c in annotations["categories"] if c["id"] == category_id][0]
        class_id = class_to_id[category_name]

        # Buscar o arquivo de imagem correspondente
        image_info = next(img for img in annotations["images"] if img["id"] == image_id)
        image_name = image_info["file_name"]

        # Caminho completo da imagem
        img_path = os.path.join(images_dir, image_name)

        # Gerar a máscara de segmentação usando RLE (Run-Length Encoding)
        rle = ann["segmentation"]  # Máscara RLE da anotação
        mask_arr = mask.decode(rle)  # Decodifica a máscara RLE em uma matriz binária

        # Criar arquivo de rótulo para cada imagem
        label_file = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(label_file, "w") as label_f:
            # Gerar a segmentação no formato YOLO (incluir a máscara em vez de uma bounding box)
            # YOLO usa segmentação com a forma de polígono (ou máscara binária) para instâncias
            # Aqui estamos criando um formato simplificado, com a máscara sendo convertida em um vetor
            polygons = mask.encode(mask_arr)
            label_f.write(f"{class_id} {polygons}\n")  # Salvando a segmentação como polígono para YOLO

"""
Caminho para o arquivo de anotações JSON do dataset BePLi
Este arquivo contém todas as anotações do dataset de plásticos nas praias
"""
annotations_file = os.path.join(extracted_dir, "annotations.json")

# Classes do dataset BePLi
classes = ['pet_bottle', 'other_bottle', 'plastic_bag', 'box_shaped_case', 'other_container', 'rope', 'other_string', 'fishing_net', 'buoy', 'other_fishing_gear', 'styrene_foam', 'others', 'fragment']

"""
Processar as anotações e salvar os rótulos para treino e validação
Isso cria os rótulos no formato necessário para o treinamento do modelo YOLO com segmentação
"""
process_annotations(annotations_file, "/content/datasets/train/images", "/content/datasets/train/labels", classes)

print("Anotações processadas e rótulos gerados com sucesso!")

"""
Agora, vamos treinar o modelo YOLOv8 com as anotações de segmentação
Esse modelo pode ser treinado com os dados processados anteriormente
"""
# Carregar o modelo YOLOv8 (você pode usar um modelo pré-treinado como "yolov8n.pt", "yolov8s.pt", etc.)
model = YOLO("yolov8n.pt")

# Iniciar o treinamento para segmentação
model.train(
    data="/content/datasets/lixo.yaml",  # Caminho do arquivo YAML
    epochs=50,
    imgsz=640,
    batch=16,
    name="garbage_segmentation_yolov8",  # Nome para os resultados do treinamento
    task="segment"  # Tarefa de segmentação
)


# Caminho da imagem a ser analisada
image_path = "/content/datasets/train/images/example_image.jpg"

# Detecção de objetos na imagem
results = model(image_path)


