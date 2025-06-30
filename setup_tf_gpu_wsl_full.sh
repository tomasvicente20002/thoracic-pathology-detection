#!/bin/bash

echo "🚀 Início da configuração do ambiente virtual com TensorFlow + GPU (compatível com CUDA 11.8)..."

# 1. Instalar python3-venv caso não esteja presente
echo "🔧 A instalar python3-venv..."
sudo apt update && sudo apt install -y python3-venv

# 2. Criar ambiente virtual
echo "📦 A criar ambiente virtual .venv..."
python3 -m venv .venv

# 3. Ativar ambiente virtual
echo "✅ A ativar ambiente virtual..."
source .venv/bin/activate

# 4. Atualizar pip e instalar TensorFlow compatível com CUDA 11.8
echo "⬆️  A atualizar pip e instalar TensorFlow 2.13..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verificar se a GPU está a ser detetada
echo "🧪 A verificar se a GPU está disponível no TensorFlow..."
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 6. Executar o script Python
echo "🚀 A executar o script chest_xray_classification.py..."
python chest_xray_classification.py

echo "✅ Tudo pronto!"
