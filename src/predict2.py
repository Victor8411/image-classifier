# src/predict2.py

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# 📍 Chemin du modèle MobileNetV2
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'mobilenet_model.h5')

# 📏 Taille d’image attendue
IMG_SIZE = (128, 128)

def predict_image(image_path):
    # Charger le modèle MobileNetV2
    model = tf.keras.models.load_model(MODEL_PATH)

    # Charger et prétraiter l’image
    image = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)[0][0]

    classe = "DOG" if prediction >= 0.5 else "CAT"
    print(f"\n🧠 Prédiction MobileNetV2 : {prediction:.2f} → {classe}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Utilisation : python src/predict2.py chemin/vers/image.jpg")
    else:
        predict_image(sys.argv[1])
