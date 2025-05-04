#!pip install ultralytics
import json
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Caminho do arquivo JSON para armazenar as predições
json_file_path = '/content/drive/My Drive/predicoes_yolov8_segmentacao.json'

# Função para salvar ou adicionar predições no JSON
def salvar_predicao(nome_imagem, predições):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            predicoes = json.load(file)
    else:
        predicoes = []

    predicoes.append({'imagem': nome_imagem, 'predicoes': predições})

    with open(json_file_path, 'w') as file:
        json.dump(predicoes, file, indent=4)

    print(f'Predição salva para {nome_imagem}')

# Função para recuperar as predições do JSON
def recuperar_predicoes():
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            predicoes = json.load(file)
        return predicoes
    else:
        return []

# Carregar o modelo YOLOv8 com segmentação
modelo = YOLO('/content/drive/My Drive/modelo_yolov8.pt')  # Caminho do modelo treinado

# Função para processar e realizar a segmentação com YOLOv8
def segmentar_objetos(imagem, modelo):
    # Realizar a detecção e segmentação com o modelo YOLOv8
    resultados = modelo(imagem)

    objetos_segmentados = []
    
    for resultado in resultados:
        for pred in resultado.boxes.data:  # boxes.data contém as predições
            classe = int(pred[5].item())  # classe do objeto detectado
            confianca = pred[4].item()  # confiança da predição
            coordenadas = pred[:4].tolist()  # coordenadas da caixa delimitadora

            # Se o modelo fizer segmentação, ele retorna as máscaras
            if 'mask' in resultado.masks:
                mascara = resultado.masks.data.cpu().numpy().tolist()
            else:
                mascara = None

            # Armazenar as informações no formato desejado
            objetos_segmentados.append({
                'classe': classe,
                'coordenadas': coordenadas,
                'confiança': confianca,
                'mascara': mascara
            })

    return objetos_segmentados

# Caminho para o diretório de imagens no Google Drive
path_imagens = '/content/drive/My Drive/imagens/'

# Loop para processar as imagens
for nome_imagem in os.listdir(path_imagens):
    if nome_imagem.endswith('.jpg') or nome_imagem.endswith('.png'):
        imagem = Image.open(os.path.join(path_imagens, nome_imagem))

        # Segmentação de objetos com o YOLOv8
        predições = segmentar_objetos(imagem, modelo)

        # Salvar as predições no JSON
        salvar_predicao(nome_imagem, predições)

# Recuperar e exibir as predições salvas
predicoes_salvas = recuperar_predicoes()
print(predicoes_salvas)
