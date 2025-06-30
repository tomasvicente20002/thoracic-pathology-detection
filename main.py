import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# --- Configuração ---
IMG_SIZE = 224
THRESHOLD = 0.5

# Lista das 14 classes do Chest X-ray14
classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

# --- Função para carregar imagem ---
def load_and_prepare_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    return np.expand_dims(img_array, axis=0)      # Batch dimension

# --- Função principal de previsão ---
def predict_image(img_path, model_path='best_model.h5'):
    if not os.path.exists(model_path):
        print(f"❌ Modelo '{model_path}' não encontrado.")
        return
    
    print(f"📦 A carregar modelo de: {model_path}")
    model = load_model(model_path)

    print(f"🖼️  A carregar imagem: {img_path}")
    img = load_and_prepare_image(img_path)

    print("🔍 A prever...")
    preds = model.predict(img)[0]  # Vetor de probabilidades

    print("\n🎯 Resultados por classe:")
    for label, prob in zip(classes, preds):
        print(f"  {label:<22}: {prob:.2f}")

    # Aplicar threshold
    predicted_labels = [label for label, prob in zip(classes, preds) if prob > THRESHOLD]

    print("\n✅ Doenças detetadas:")
    if predicted_labels:
        for label in predicted_labels:
            print(f"  - {label}")
    else:
        print("  Nenhuma com probabilidade > 0.5")

# --- Execução via linha de comandos ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classificação de radiografias com DenseNet121")
    parser.add_argument('--image_path', type=str, default='C:\\Users\\lynxv\\Desktop\\Faculdade\\Mestrado\\2º Semestre\APVC\\Project\\data_set\\images_006\\images\\00011558_008.png',  help='Caminho para a imagem .png a prever')
    parser.add_argument('--model', type=str, default='C:\\Users\\lynxv\\Desktop\\Faculdade\\Mestrado\\2º Semestre\APVC\\Project\\best_model.h5', help='Modelo Keras treinado (.h5)')
    args = parser.parse_args()
    
    predict_image(args.image_path, args.model)
