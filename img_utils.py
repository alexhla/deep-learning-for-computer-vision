# Dependency Installation (Linux):
# $ sudo apt install python3-numpy
# $ pip3 install argparse opencv-python opencv-contrib-python

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# instantiate the argument parser object
ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument('-d', '--directory', required=True, help='path to a directory of images (required)')
ap.add_argument('-di', '--directory_info', action='store_true', help='get info on images in directory')
ap.add_argument('-dd', '--delete_duplicates', action='store_true', help='find and delete duplicate images')
ap.add_argument('-ii', '--image_info', help='get info on image at given index')
ap.add_argument('-r_stretch', '--resize_with_stretching', nargs=3, help='resize image at a given index to specified dimensions')
ap.add_argument('-r_padding', '--resize_with_padding', nargs=2, help='resize image at a given index to specified dimensions maintaining aspect ratio')
ap.add_argument('-s', '--show', help='show the image at the given index')
ap.add_argument('-sbgr', '--show_bgr', help='show rgb channels of the image at the given index')
ap.add_argument('-scym', '--show_cym', help='show cyan (bg), yellow (gr), magenta (rb) channels of the image at the given index')
ap.add_argument('-a', '--add', nargs=2, help='add two images at the given indices')



args = vars(ap.parse_args())
files = os.listdir(args['directory'])
print(f'argparse arguments: {args}')
print(f'number of images found: {len(files)}')

if args['resize_with_padding']:
	index = args['resize_with_padding'][0]
	desired_size = args['resize_with_padding'][1]
	print(f'resizing image at index {index} to long side pixel dimensions {desired_size} with padding on short side')

	

	filename = files[int(index)]
	im = cv2.imread(args['directory'] + filename)

	old_size = im.shape[:2] # old_size is in (height, width) format
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])


	print(f'pixel new_size is {new_size}')




	cv2.imshow('', im)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
# crop to multiple of 256

	
# resize to 256



if args['image_info']:
	index = args['image_info'][0]
	filename = files[int(index)]
	im = cv2.imread(args['directory'] + filename)
	print(f'\nFile: {filename} {type(im)}')
	print(f'Image dtype: {im.dtype}')
	print(f'Image size: {im.size} (total pixels)')	
	print(f'Image Shape: {im.shape}')
	print(f'Image Height: {im.shape[0]}')
	print(f'Image Width: {im.shape[1]}')
	print(f'Number of Channels: {im.shape[2]}')
	cv2.imshow('', im)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()


if args['add']:
	print(args['add'])
	im1 = cv2.imread(args['directory'] + files[int(args['add'][0])])
	im2 = cv2.imread(args['directory'] + files[int(args['add'][1])])
	h = min(im1.shape[0], im2.shape[0])
	w =  min(im1.shape[1], im2.shape[1])
	im1_cropped = im1[0:h, 0:w]  # crop to match image dimensions, preserving aspect ratio
	im2_cropped = im2[0:h, 0:w]
	im_sum = cv2.add(im1_cropped, im2_cropped)
	cv2.imshow('', im_sum)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()


if args['show']:
	im = cv2.imread(args['directory'] + files[int(args['show'])])
	cv2.imshow('', im)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()

if args['show_bgr']:
	im = cv2.imread(args['directory'] + files[int(args['show_bgr'])])
	b,g,r = cv2.split(im)
	z = np.zeros(b.shape, dtype=np.uint8)
	b = cv2.merge((b, g, z))
	g = cv2.merge((z, g, z))
	r = cv2.merge((z, z, r))
	cv2.imshow("im-b", b)
	cv2.imshow("im-g", g)
	cv2.imshow("im-r", r)
	cv2.waitKey(5000)  
	cv2.destroyAllWindows()  

if args['show_cym']:
	im = cv2.imread(args['directory'] + files[int(args['show_cym'])])
	b,g,r = cv2.split(im)
	z = np.zeros(b.shape, dtype=np.uint8)
	bg = cv2.merge((b, g, z))
	gr = cv2.merge((z, g, r))
	rb = cv2.merge((b, z, r))
	cv2.imshow("im-bg", bg)
	cv2.imshow("im-gr", gr)
	cv2.imshow("im-rb", rb)
	cv2.waitKey(5000)  
	cv2.destroyAllWindows()  


if args['directory_info']:
	image_count_by_shape = {}
	image_count_by_height = {}
	image_count_by_width = {}
	image_count_by_channels = {}

	for index, file in enumerate(files):

		if index > 10:  # batch size for testing
			cv2.imshow("im",im)
			cv2.waitKey(5000)  
			cv2.destroyAllWindows()  
			break

		im = cv2.imread(args["directory"] + file)

		height = im.shape[0]
		width = im.shape[1]
		channels = im.shape[2]

		print(f'\nFile: {file} {type(im)}')
		print(f'Image Shape: {im.shape}')		
		print(f'Image Height: {height}')
		print(f'Image Width: {width}')
		print(f'Number of Channels: {channels}')

		# tally images by im.shape
		if im.shape in image_count_by_shape:
			image_count_by_shape[im.shape] += 1
		else:
			image_count_by_shape[im.shape] = 1

		# tally images by width
		if width in image_count_by_width:
			image_count_by_width[width] += 1
		else:
			image_count_by_width[width] = 1

		# tally images by height
		if height in image_count_by_height:
			image_count_by_height[height] += 1
		else:
			image_count_by_height[height] = 1

		# tally images by channels
		if channels in image_count_by_channels:
			image_count_by_channels[channels] += 1
		else:
			image_count_by_channels[channels] = 1

	# print(f'\nImage Count by Pixel Dimensions: {image_count_by_shape}')
	# print(f'\nImage Count by Pixel Width: {image_count_by_width}')
	# print(f'\nImage Count by Pixel Height: {image_count_by_height}')
	# print(f'\nImage Count by Pixel Channels: {image_count_by_channels}')

	image_count_by_sorted_width = dict(sorted(image_count_by_width.items()))
	image_count_by_sorted_height = dict(sorted(image_count_by_height.items()))

	# print(f'\nImage Count by Pixel Height Sorted: {image_count_by_sorted_width}')
	# print(f'\nImage Count by Pixel Width Sorted: {image_count_by_sorted_height}')

	width_keys = [str(key) for key in image_count_by_sorted_width.keys()]
	height_keys = [str(key) for key in image_count_by_sorted_height.keys()]

	width_values = image_count_by_sorted_width.values()
	height_values = image_count_by_sorted_height.values()

	fig = plt.figure(figsize = (12.8, 7.2))
	fig.suptitle(f'Image Distribution by Resolution'
				+f'\nDirectory: {args["directory"]}'
				+f'\nNumber of Images: {len(files)}')
	
	ax1 = fig.add_subplot(211)
	ax1.set_ylabel('Files per Pixel Width')
	ax1.set_xticks(np.linspace(0, len(width_keys)-1, 10))
	ax1.bar(width_keys, width_values, color = 'seagreen')
	
	ax2 = fig.add_subplot(212)	
	ax2.set_ylabel('Files per Pixel Height')
	ax2.set_xticks(np.linspace(0, len(height_keys)-1, 10))
	ax2.bar(height_keys, height_values, color = 'steelblue')
	
	plt.savefig('plots/image_distribution_by_resolution_'+ f'{args["directory"].replace("/","_")[:-1]}' + '.png')
	plt.show()


if args['delete_duplicates']:

	for index, file in enumerate(files):
		
		im = cv2.imread(args["directory"] + file)
		
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		hsh = cv2.img_hash.BlockMeanHash_create()
		hsh.compute(gray)
		print(f'hash: {hsh}')

		if index > 10:  # batch size for testing
			cv2.imshow("gray",gray)
			cv2.waitKey(5000)  
			cv2.destroyAllWindows()
			break