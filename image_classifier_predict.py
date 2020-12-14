import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

DIRECTORY_DATASET_OUTER = 'img/paintings/artist/paul_cezanne/256x256/'
DIRECTORY_DATASET_INNER = 'img/paintings/artist/paul_cezanne/256x256/256x256/'
DIRECTORY_SOURCE = 'img/dataset/'
DIRECTORY_CLASS_A = 'img/drawings/artist/paul_cezanne/original/'
DIRECTORY_CLASS_B = 'img/paintings/artist/paul_cezanne/original/'
DIRECTORY_DESTINATION = ''

image_size = (256, 256)
batch_size = 32

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	DIRECTORY_DATASET_OUTER,
	shuffle=False,
	image_size=image_size,
	batch_size=batch_size,
)

val_ds = val_ds.prefetch(buffer_size=32)
loaded_model = keras.models.load_model('models/paintings_vs_drawings_classification_model_25_epochs')
predictions = loaded_model.predict(val_ds)

files = os.listdir(DIRECTORY_DATASET_INNER)
files.sort()

for index, filename in enumerate(files):

	if round(np.asscalar(predictions[index]), 2) <= 0.5:
		DIRECTORY_DESTINATION = DIRECTORY_CLASS_A
	else:
		DIRECTORY_DESTINATION = DIRECTORY_CLASS_B

	print(f'Moving {filename} to {DIRECTORY_DESTINATION}')
	os.replace(os.path.join(DIRECTORY_SOURCE, filename), os.path.join(DIRECTORY_DESTINATION, filename))