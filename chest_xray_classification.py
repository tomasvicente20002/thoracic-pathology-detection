# chest_xray_classification.py
# Classificação de doenças pulmonares com DenseNet121 e Chest X-ray14

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


# --- 1. Carregar e preparar dados ---
CSV_PATH = 'data_set/Data_Entry_2017.csv'

# Carregar CSV
df = pd.read_csv(CSV_PATH)
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

# Binarizar etiquetas
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['Finding Labels'])
classes = mlb.classes_

for i, class_name in enumerate(classes):
    df[class_name] = labels[:, i]

# Procurar imagens apenas nas 3 pastas específicas
subdirs = [
    "data_set/images_001/images",
    "data_set/images_002/images",
    "data_set/images_003/images",
    "data_set/images_004/images",
    "data_set/images_005/images",
    "data_set/images_006/images",
    "data_set/images_007/images",
    "data_set/images_008/images",
    "data_set/images_009/images",
    "data_set/images_0010/images",
    "data_set/images_0011/images",
    "data_set/images_0012/images"
]
image_paths = []
for subdir in subdirs:
    image_paths.extend(glob(f"{subdir}/*.png"))

# Mapear nome da imagem → caminho completo
image_map = {os.path.basename(p): p for p in image_paths}
df['path'] = df['Image Index'].map(image_map)

# Manter apenas entradas válidas
df = df[df['path'].notnull()]

# Subconjunto para treino rápido
df_small = df.sample(n=2000, random_state=42)
train_df, val_df = train_test_split(df_small, test_size=0.2, random_state=42)

# --- 2. Geradores de imagens ---
IMG_SIZE = 224
BATCH_SIZE = 32

train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=10,
                               zoom_range=0.1,
                               horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

def make_generator(dataframe, gen):
    return gen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='path',
        y_col=classes.tolist(),
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

train_generator = make_generator(train_df, train_gen)
val_generator = make_generator(val_df, val_gen)

# --- 3. Modelo CNN com DenseNet121 ---
base_model = DenseNet121(weights='imagenet', include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
out = Dense(len(classes), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Callbacks ---
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True,
                             monitor='val_loss', mode='min')
earlystop = EarlyStopping(patience=5, restore_best_weights=True)

# --- 5. Treino ---
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=15,
                    callbacks=[checkpoint, earlystop])

# --- 6. Avaliação ---
model.evaluate(val_generator)

model.save("best_model_saved.h5")

# --- 7. Gráfico de loss ---
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Binary Crossentropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()
