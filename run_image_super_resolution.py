import os
import time
import numpy as np
from PIL import Image
from ISR.models import RDN

print(os.path.dirname(os.path.realpath(__file__)))

INPUT_PATH = '/home/alexander/Documents/deep-learning-for-computer-vision/lib/image-super-resolution/img_in/'
OUTPUT_PATH = '/home/alexander/Documents/deep-learning-for-computer-vision/lib/image-super-resolution/img_out/'

for file in os.listdir(INPUT_PATH):
	print(file)

	tic = time.time()

	img = Image.open(os.path.join(INPUT_PATH, file))
	lr_img = np.array(img)


	rdn = RDN(weights='psnr-small')

	#sr_img = rdn.predict(lr_img)
	sr_img = rdn.predict(lr_img, by_patch_of_size=50)
	im1 = Image.fromarray(sr_img)

	print(os.path.join(OUTPUT_PATH, file))
	im1.save(os.path.join(OUTPUT_PATH, file))


	toc = time.time()
	print('Elapsed Time:')
	print(round((toc - tic)/60, 2))