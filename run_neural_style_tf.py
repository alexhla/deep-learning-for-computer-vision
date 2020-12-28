import os
import time
import subprocess

LIB = 'neural_style.py'
LIB_PATH = '../neural-style-tf-master/'


# for index, style_img in enumerate(os.listdir(os.path.join(LIB_PATH, 'styles'))):
# 	print(f'--------------- {index} - {style_img} ---------------')

for index, content_img in enumerate(os.listdir(os.path.join(LIB_PATH, 'image_input'))):
	print(f'--------------- {content_img} ---------------')


	for style_img in os.listdir(os.path.join(LIB_PATH, 'styles')):
		print(f'\n{style_img}')

	# for content_img in os.listdir(os.path.join(LIB_PATH, 'image_input')):
	# 	print(f'\n{content_img}')
		output_img =  str(index) + ' - ' + style_img[:-4] + ' - ' +content_img[:-4]
		# output_pixel_max = 512
		output_pixel_max = 1280
		print(f'output pixel max: {output_pixel_max}')

		tic = time.time()

		subprocess.run(['python', os.path.join(LIB_PATH, LIB),
			'--style_imgs', style_img,
			'--content_img', content_img,
			'--img_name', output_img,
			'--max_size', str(output_pixel_max)],
			capture_output=True, cwd=LIB_PATH)

		toc = time.time()
		print(f'Elapsed time is {round((toc - tic)/60, 2)} minutes')