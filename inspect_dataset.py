import os
import json
import random
import numpy as np
import cv2
from PIL import Image

CWD = os.getcwd()
ORIGINAL_DATA_PATH = f'{CWD}/original_dataset'

def find_largest_image():
    largest_width = 0
    height = 0

    for filename in os.listdir(f'{CWD}/original_dataset/'):
        image = Image.open(f'{CWD}/original_dataset/{filename}')
        
        if image.width > largest_width:
            largest_width = image.width
            height = image.height
    
    return (largest_width, height)


def find_smallest_image():
    smallest_width = float('Inf')
    height = 0

    for filename in os.listdir(f'{CWD}/original_dataset/'):
        image = Image.open(f'{CWD}/original_dataset/{filename}')
        
        if image.width < smallest_width:
            smallest_width = image.width
            height = image.height
    
    return (smallest_width, height)


def group_images_by_classname():
    images = {}

    for filename in os.listdir(f'{CWD}/original_dataset/'):
        if not 'r' in filename:
            continue
            
        classname = filename.split('_')[0]

        try:
            images[classname].append(filename)
        except KeyError:
            images[classname] = [filename]

    return images


def write_noisy_image(filename, output_name):
    import skimage
    from skimage import io
    image = skimage.io.imread(filename)/255.0
    image = skimage.util.random_noise(image, mode='gaussian', mean=0.01)*255
    skimage.io.imsave(output_name, image.astype(np.uint8))
    return image


# Find the smallest and largest image in the set
w, h = find_largest_image()
print(f'Largest image: {w}x{h}px')
w, h = find_smallest_image()
print(f'Smallest image: {w}x{h}px')

# Count how many images each classname has
images = group_images_by_classname()
image_count = {}

for k, v in images.items():
    try:
        image_count[len(v)] += 1
    except KeyError:
        image_count[len(v)] = 1

print(json.dumps(image_count, indent=2))

image_directory = f'{CWD}/original_dataset/'
filename = f'{image_directory}/{random.choice(os.listdir(image_directory))}'
output_name = f'{CWD}/noisy_image.jpg'
write_noisy_image(filename, output_name)
