import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

np.set_printoptions(suppress=True)  # suppress exponential notation

image_size = (256, 256)
batch_size = 32


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"img/test/",
	shuffle=False,
	image_size=image_size,
	batch_size=batch_size,
)


val_ds = val_ds.prefetch(buffer_size=32)

loaded_model = keras.models.load_model('models/paintings_vs_drawings_classification_model_25_epochs')

predictions = loaded_model.predict(val_ds)



print(predictions)



for (root,dirs,files) in os.walk('img/test/'): 
	print (root) 
	print (dirs) 
	print (files) 
	print ('--------------------------------') 



f = os.listdir('img/test/a')
f.sort()
print(f)