# 🧠 YOLOv8 - Segmentação Automatizada com Imagens do Google Drive

Este projeto executa automaticamente um modelo YOLOv8 com **segmentação** em imagens armazenadas no Google Drive, utilizando o **Google Colab**. As predições são salvas em um arquivo JSON, com possibilidade de filtro por confiança mínima.

## 🚀 Funcionalidades

- 📂 Integração com Google Drive para buscar imagens automaticamente.
- 🧠 Execução do modelo **YOLOv8 com segmentação**.
- ✅ Filtragem por confiança (threshold ajustável).
- 📄 Geração e atualização de um JSON com os resultados.
- 🔁 Recuperação dos dados a partir do JSON.
- 🔧 Pronto para rodar no **Google Colab**.

---

## 📁 Estrutura Esperada

📁 MeuDrive/
└── 📁 imagens/
├── img1.jpg
├── img2.jpg
└── ...

---

## ⚙️ Como Usar

### 1. Clone este repositório ou copie o notebook para o Google Colab

```bash
git clone https://github.com/seuusuario/yolov8-segmentacao-colab.git

```

### 2. Monte o Google Drive no Colab

```bash
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Carregue o modelo YOLOv8

```bash
from ultralytics import YOLO
modelo = YOLO('yolov8n-seg.pt') # ou yolov8s-seg.pt, yolov8m-seg.pt etc.
```

### 4. Execute a inferência nas imagens

```bash
processar_imagens(
pasta_imagens='/content/drive/MyDrive/imagens/',
modelo=modelo,
json_saida='resultados.json',
confianca_threshold=0.8
)
```

### 5. Recuperar os dados do JSON

```json
dados = carregar_json('resultados.json')

# Exemplo de Estrutura do JSON
[
    {
        "imagem": "img1.jpg",
        "predicoes": [
            {
                "classe": 0,
                "coordenadas": [100, 50, 300, 200],
                "confiança": 0.92,
                "mascara": [[...]]
            }
        ]
    }
]
```

### 🛠️ Funções Principais

```bash
processar_imagens() Processa todas as imagens da pasta e salva/atualiza JSON
carregar_json() Lê e retorna o conteúdo do JSON
```

✅ Requisitos

```bash
Python 3.8+

Google Colab

Ultralytics

YOLOv8 com modelo .pt de segmentação (yolov8n-seg.pt, etc.)
```

📝 Observações
Apenas predições com confiança superior ao threshold (por padrão, 80%) são salvas.

A máscara é opcional e pode ser desabilitada conforme seu uso.
