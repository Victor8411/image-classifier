import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Chemin du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')

# Fonction de prédiction
def predict_image(image_path):
    # Charger le modèle
    model = tf.keras.models.load_model(MODEL_PATH)

    # Charger et prétraiter l’image
    image = Image.open(image_path).resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch size = 1

    # Prédire
    prediction = model.predict(img_array)[0][0]

    # Afficher le résultat
    classe = "DOG"  if prediction >= 0.5 else "CAT"
    print(f"\nPrédiction : {prediction:.2f} → {classe}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Utilisation : python src/predict.py chemin/vers/image.jpg")
    else:
        predict_image(sys.argv[1])




