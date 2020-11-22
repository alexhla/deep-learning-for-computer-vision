import os
import argparse
from PIL import Image

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
	number_of_files_in_directory = len(files)
	print(f'\nDirectory: {args["directory"]}\n')
	print(f'Number of Images in Directory:\t{number_of_files_in_directory}')
	
	image_count_by_format = {}
	image_count_by_mode = {}
	image_count_by_size = {}
	image_max_width = 0
	image_max_height = 0
	image_min_width = 10000
	image_min_height = 10000

	for file in files:
		im = Image.open(args["directory"] + file)

		# count images by format
		if im.format in image_count_by_format:
			image_count_by_format[im.format] += 1
		else:
			image_count_by_format[im.format] = 1

		# count images by mode
		if im.mode in image_count_by_mode:
			image_count_by_mode[im.mode] += 1
		else:
			image_count_by_mode[im.mode] = 1

		# count images by size
		if im.size in image_count_by_size:
			image_count_by_size[im.size] += 1
		else:
			image_count_by_size[im.size] = 1

		# get max / min
		width, height = im.size
		image_max_width = max(image_max_width, width)
		image_max_height = max(image_max_height, height)
		image_min_width = min(image_min_width, width)
		image_min_height = min(image_min_height, height)

	print(f'Image Count by Format: {image_count_by_format}')
	print(f'Image Count by Mode: {image_count_by_mode}')
	print(f'Image Pixel Max (w,h): ({image_max_width},{image_max_height})')
	print(f'Image Pixel Min (w,h): ({image_min_width},{image_min_height})')
	# print(f'Image Count by Pixel Resolution: {image_count_by_size}')


if args['command'] == 'find_duplicate_images':
	print(f'finding duplicates')