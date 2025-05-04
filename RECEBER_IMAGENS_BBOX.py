import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# Caminho do arquivo JSON
json_file_path = '/content/drive/My Drive/predicoes_yolo.json'

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

# Função para processar e realizar a detecção com o modelo YOLO
def detectar_objetos(imagem, modelo):
    # Ajustar conforme a entrada do seu modelo
    imagem_redimensionada = imagem.resize((416, 416))  # Supondo que o YOLO use 416x416
    imagem_array = np.array(imagem_redimensionada) / 255.0
    imagem_array = np.expand_dims(imagem_array, axis=0)

    # Realizar a detecção
    predições = modelo.predict(imagem_array)

    # Processar as predições do YOLO (ajuste conforme o formato da saída)
    objetos_detectados = []
    for pred in predições[0]:  # Supondo que a predição seja uma lista de caixas
        if pred[4] > 0.5:  # Threshold de confiança (ajuste conforme necessário)
            classe = int(pred[5])
            coordenadas = pred[:4]  # [x_min, y_min, x_max, y_max]
            objetos_detectados.append({
                'classe': classe,
                'coordenadas': coordenadas,
                'confiança': pred[4]
            })
    
    return objetos_detectados

# Caminho para o diretório de imagens no Google Drive
path_imagens = '/content/drive/My Drive/imagens/'

# Loop para processar as imagens
for nome_imagem in os.listdir(path_imagens):
    if nome_imagem.endswith('.jpg') or nome_imagem.endswith('.png'):
        imagem = Image.open(os.path.join(path_imagens, nome_imagem))

        # Detecção de objetos com o YOLO
        predições = detectar_objetos(imagem, modelo)

        # Salvar as predições no JSON
        salvar_predicao(nome_imagem, predições)

# Recuperar e exibir as predições salvas
predicoes_salvas = recuperar_predicoes()
print(predicoes_salvas)
