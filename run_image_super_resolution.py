import os
import time
import numpy as np
from PIL import Image
from ISR.models import RDN

print(os.path.dirname(os.path.realpath(__file__)))

INPUT_PATH = 'img/img_in/'
OUTPUT_PATH = 'img/img_out/'

for file in os.listdir(INPUT_PATH):
	print(file)

	tic = time.time()

	img = Image.open(os.path.join(INPUT_PATH, file))
	lr_img = np.array(img)


	rdn = RDN(weights='psnr-small')

	#sr_img = rdn.predict(lr_img)
	sr_img = rdn.predict(lr_img, by_patch_of_size=50)
	im1 = Image.fromarray(sr_img)

	im1.save(os.path.join(OUTPUT_PATH, file))


	toc = time.time()
	print('Elapsed Time:')
	print(round((toc - tic)/60, 2))