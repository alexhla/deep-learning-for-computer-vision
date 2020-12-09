import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2


image_size = (256, 256)
batch_size = 32


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"img/test/",
	seed=4321,
	image_size=image_size,
	batch_size=batch_size,
)



val_ds = val_ds.prefetch(buffer_size=32)


loaded_model = keras.models.load_model('mymodel')

predictions = loaded_model.predict(val_ds)



print(predictions)