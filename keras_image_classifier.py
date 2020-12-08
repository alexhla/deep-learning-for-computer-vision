import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Generate The Dataset

image_size = (256, 256)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"img/paintings_vs_drawings/",
	validation_split=0.2,
	subset="training",
	seed=4321,
	image_size=image_size,
	batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"img/paintings_vs_drawings/",
	validation_split=0.2,
	subset="validation",
	seed=4321,
	image_size=image_size,
	batch_size=batch_size,
)


# Visulalize Training Dataset

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
# 	for i in range(9):
# 		ax = plt.subplot(3, 3, i + 1)
# 		plt.imshow(images[i].numpy().astype("uint8"))
# 		plt.title(int(labels[i]))
# 		plt.axis("off")
# 	plt.show()


data_augmentation = keras.Sequential(
	[
		layers.experimental.preprocessing.RandomFlip("horizontal"),
		layers.experimental.preprocessing.RandomRotation(0.1),
	]
)


# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
# 	for i in range(9):
# 		augmented_images = data_augmentation(images)
# 		ax = plt.subplot(3, 3, i + 1)
# 		plt.imshow(augmented_images[0].numpy().astype("uint8"))
# 		plt.axis("off")
# 	plt.show()


train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
	inputs = keras.Input(shape=input_shape)
	# Image augmentation block
	x = data_augmentation(inputs)

	# Entry block
	x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
	x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)

	x = layers.Conv2D(64, 3, padding="same")(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)

	previous_block_activation = x  # Set aside residual

	for size in [128, 256, 512, 728]:
		x = layers.Activation("relu")(x)
		x = layers.SeparableConv2D(size, 3, padding="same")(x)
		x = layers.BatchNormalization()(x)

		x = layers.Activation("relu")(x)
		x = layers.SeparableConv2D(size, 3, padding="same")(x)
		x = layers.BatchNormalization()(x)

		x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

		# Project residual
		residual = layers.Conv2D(size, 1, strides=2, padding="same")(
			previous_block_activation
		)
		x = layers.add([x, residual])  # Add back residual
		previous_block_activation = x  # Set aside next residual

	x = layers.SeparableConv2D(1024, 3, padding="same")(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)

	x = layers.GlobalAveragePooling2D()(x)
	if num_classes == 2:
		activation = "sigmoid"
		units = 1
	else:
		activation = "softmax"
		units = num_classes

	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(units, activation=activation)(x)
	return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)



## Train The Model

epochs = 50

callbacks = [
	keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
	optimizer=keras.optimizers.Adam(1e-3),
	loss="binary_crossentropy",
	metrics=["accuracy"],
)
model.fit(
	train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
