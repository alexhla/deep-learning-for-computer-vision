import os
import time
import argparse
import numpy as np
from PIL import Image
from ISR.models import RDN

# Instantiate the argument parser object
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument('-i', '--image_path', nargs=1, metavar=('IMAGE PATH'), required=True, help='path to image (required)')
ap.add_argument('-x', '--resolution_doubler', nargs=1, metavar=('RESOLUTION DOUBLER'), required=True, help='number of times to double (2x) the resolution (required)')

args = vars(ap.parse_args())
IMAGE_PATH = args['image_path'][0]
RESOLUTION_DOUBLER = args['resolution_doubler'][0]
current_resolution = ''

print('argparse arguments: %s\n' % args)
print('Image --- %s' % IMAGE_PATH)
print('Doubler --- %sx' % RESOLUTION_DOUBLER)

for i in range(0, int(RESOLUTION_DOUBLER)):
	tic = time.time()

	current_image_path = IMAGE_PATH[:-4] + current_resolution + '.png'
	print('Current Image', current_image_path)
	img = Image.open(os.path.join(current_image_path))

	lr_img = np.array(img)
	rdn = RDN(weights='psnr-small')
	sr_img = rdn.predict(lr_img, by_patch_of_size=50)
	img_doubled = Image.fromarray(sr_img)

	current_resolution = str(max(img_doubled.size))
	new_image_path = IMAGE_PATH[:-4] + current_resolution + '.png'
	img_doubled.save(new_image_path)

	current_image_path = new_image_path

	toc = time.time()
	print('Elapsed Time:')
	print(round((toc - tic)/60, 2))