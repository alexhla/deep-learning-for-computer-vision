import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# instantiate the argument parser object
ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument("-d", "--directory", required=True, help="The directory containing images of concern")
ap.add_argument("-c", "--command", required=True, help="The operation to be done on the directory of concern")
args = vars(ap.parse_args())
# print(args)

available_commands = ['get_info', 'find_duplicate_images']
if args['command'] not in available_commands:
	print(f'\nError: Command \"{args["command"]}\" Not Found\n')


if args['command'] == 'get_info':
	files = os.listdir(args['directory'])

	image_count_by_dimensions = {}	
	image_count_by_height = {}
	image_count_by_width = {}
	image_count_by_channels = {}

	for index, file in enumerate(files):

		if index > 50:  # batch size for testing
			break

		im = cv2.imread(args["directory"] + file)

		dimensions = im.shape
		height = im.shape[0]
		width = im.shape[1]
		channels = im.shape[2]

		print(f'\nFile: {file} {type(im)}')
		print(f'Image Dimensions: {dimensions}')
		print(f'Image Height: {height}')
		print(f'Image Width: {width}')
		print(f'Number of Channels: {channels}')

		# tally images by dimensions
		if dimensions in image_count_by_dimensions:
			image_count_by_dimensions[dimensions] += 1
		else:
			image_count_by_dimensions[dimensions] = 1

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

	# print(f'\nImage Count by Pixel Dimensions: {image_count_by_dimensions}')
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

	fig = plt.figure(figsize=(12,8))
	fig.suptitle(f'Image Distribution by Resolution'
				+f'\nDirectory: {args["directory"]}'
				+f'\nNumber of Images: {len(files)}')
	
	ax1 = fig.add_subplot(211)
	ax1.set_ylabel('Files per Pixel Width')
	ax1.set_xticks(np.linspace(0, len(width_keys)-1, 10))
	ax1.bar(width_keys, width_values, color='seagreen')
	
	ax2 = fig.add_subplot(212)	
	ax2.set_ylabel('Files per Pixel Height')
	ax2.set_xticks(np.linspace(0, len(height_keys)-1, 10))
	ax2.bar(height_keys, height_values, color='steelblue')
	
	plt.savefig('plots/image_distribution_by_resolution_'+ f'{args["directory"].replace("/","_")[:-1]}' + '.png')
	plt.show()


if args['command'] == 'find_duplicates':
	print(f'finding duplicates')



if args['command'] == 'resize256':
	print(f'resize256')
# resize

# crop to multiple of 256

# resize to 256

