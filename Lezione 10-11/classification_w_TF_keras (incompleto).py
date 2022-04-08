# obiettivi
# 1 - caricare set di dati da disco
# 2 - identificare overfitting e miliorarlo (data augmentation e dropout)
# 3 - esaminare e comprendere i dati
# 4 - costruire modello
# 5 - allenare modello
# 6 - testare modello
# 7 - migliorare modello

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from keras import layers
from keras.models import Sequential

# importiamo dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# impostiamo alcuni valori
batch_size = 32
img_height = 180
img_width = 180

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 80% img per formazione, 20% per convalida
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 80% img per formazione, 20% per convalida
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#visualizziamo alcuni dati

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(labels[i])
        plt.axis("off")


