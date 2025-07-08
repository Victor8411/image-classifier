import os
import pytest
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#  Dossiers avec quelques images pour test rapide
TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')
VAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'val')
IMG_SIZE = (64, 64)
BATCH_SIZE = 8

def test_cnn_accuracy_above_threshold():
    #  Skip le test si les dossiers n'existent pas (utile sur GitHub Actions)
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        pytest.skip("Dossiers 'data/train' ou 'data/val' introuvables, test ignoré")

    #  Préparation des données
    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    val_data = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    #  Modèle mini-CNN pour test rapide
    model = models.Sequential([
        layers.Conv2D(8, (3,3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #  Entraînement rapide (1 epoch)
    history = model.fit(train_data, epochs=1, validation_data=val_data, verbose=0)
    val_accuracy = history.history['val_accuracy'][0]

    print(f" Accuracy obtenue sur val : {val_accuracy:.2f}")
    assert val_accuracy > 0.5, f" Accuracy trop basse : {val_accuracy:.2f}"



