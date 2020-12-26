import os

SOURCE_DIR_PATH = '../neural-style-tf-master/image_output/'
NEW_DIR_PATH = '../styled_photos/'


### MOVE Files

for folder in os.listdir(SOURCE_DIR_PATH):

	files = os.listdir(os.path.join(SOURCE_DIR_PATH, folder))

	for file in files:

		if file[0:1].isdigit():

			print(f'{file}')

			x = os.path.join(SOURCE_DIR_PATH, folder, file)
			y = os.path.join(NEW_DIR_PATH, file)

			os.rename(x, y)



