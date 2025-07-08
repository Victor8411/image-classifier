import tensorflow_datasets as tfds
import tensorflow as tf
import os

# Chemins des dossiers
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Téléchargement du dataset (chat vs chien)
(ds_train, ds_val), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

# Fonction pour enregistrer les images
def save_images(dataset, target_dir, max_images=100):
    for label in [0, 1]:
        class_name = "cat" if label == 0 else "dog"
        class_path = os.path.join(target_dir, class_name)
        os.makedirs(class_path, exist_ok=True)

    counter = {0: 0, 1: 0}
    for image, label in dataset:
        if counter[int(label)] >= max_images:
            continue
        image = tf.image.resize(image, (64, 64))
        label_name = "cat" if label == 0 else "dog"
        path = os.path.join(target_dir, label_name, f"{label_name}_{counter[int(label)]}.jpg")
        tf.keras.utils.save_img(path, image)
        counter[int(label)] += 1
        if sum(counter.values()) >= max_images * 2:
            break

# Enregistrement
print("📥 Téléchargement et génération des images...")
save_images(ds_train, TRAIN_DIR, max_images=100)
save_images(ds_val, VAL_DIR, max_images=20)
print("✅ Données enregistrées dans 'data/train' et 'data/val'")



