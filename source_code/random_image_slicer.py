# For Making Image Patch

import os
import random
from PIL import Image

origin_data_path = ".\\data" # Source Data Path
result_data_path = ".\\result" # Sliced Data Save Path

image_size = 256 # Square Image
patch_size = 128
patch_nums = 10

if len(os.listdir(result_data_path)) == 0:
	
	for i in range(len(os.listdir(origin_data_path))):

		os.mkdir(os.path.join(result_data_path, str(i)))


for (subdirs, dirs, files) in os.walk(origin_data_path):

	# Exclude orgin dir path
	if subdirs == origin_data_path:
		continue

	print(subdirs)

	for index, name in enumerate(os.listdir(subdirs)):

		img_name = os.path.join(subdirs,name[:len(name)-4])

		img = Image.open(os.path.join(subdirs,name))

		for i in range(1,patch_nums + 1):
			
			rand_x = random.randrange(0,image_size-patch_size)
			rand_y = random.randrange(0,image_size-patch_size)

			img_crop = img.crop((rand_x, rand_y, rand_x + patch_size, rand_y + patch_size))

			img_name = img_name.replace("data","result")

			img_crop_name = img_name + "_{:03d}".format(int(i)) + ".tif"

			print(img_crop_name, img_crop.size)

			img_crop.save(img_crop_name)

