import os

SOURCE_DIR_PATH = 'lib/neural-style-tf-master/image_output/'
NEW_DIR_PATH = 'img/etsy/styled_photos/'


### MOVE Files

for folder in os.listdir(SOURCE_DIR_PATH):

	files = os.listdir(os.path.join(SOURCE_DIR_PATH, folder))

	for file in files:

		if len(file) > 20:

			print(f'{file}')

			x = os.path.join(SOURCE_DIR_PATH, folder, file)
			y = os.path.join(NEW_DIR_PATH, file)

			os.rename(x, y)



