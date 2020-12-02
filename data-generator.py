import os
from shutil import copyfile
from PIL import Image
import torch
import torchvision.transforms as transforms
import skimage
import numpy
import cv2
from skimage import io
from tqdm import tqdm
import threading
import sys
import numpy as np
import random

orig_imgs = os.getcwd() + "/wrasse/"
testing_data = "./testing_data/"
training_data = "./training_data/"

images = os.listdir(orig_imgs)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


try:
    os.mkdir(testing_data)
except FileExistsError:
    pass
try:
    os.mkdir(training_data)
except FileExistsError:
    pass

for image in images:
    if image.__contains__('r'):
        image_id = image.split('_')[0]
        try:
            os.mkdir(testing_data + image_id)
        except FileExistsError:
            pass
        try:
            os.mkdir(training_data + image_id)
        except FileExistsError:
            pass

        if int(image.split('_')[2].split('.')[0]) == 1:
            copyfile(orig_imgs + image, testing_data + image_id + '/' + image)
        else:
            copyfile(orig_imgs + image, training_data + image_id + '/' + image)

root_dir = './training_data/'
categories = [(folder, [image for image in os.listdir(root_dir + "/" + folder)]) for folder in os.listdir(root_dir)]

def create_image(img, num):
    # Open image once
    open_image = Image.open(img)

    # List of transformations done
    transformations = [
        ("Rotate", transforms.RandomRotation(180, resample=Image.BILINEAR)),
        ("Contrast", transforms.ColorJitter(contrast=0.5)),
        ("Affine", transforms.RandomAffine(degrees=0, translate=(0, 0.1))),
        ("Rotate_Contrast", transforms.Compose([
            transforms.ColorJitter(contrast=0.3),
            transforms.RandomAffine(degrees=180, translate=(0, 0.1))
        ])),
        ("Random_Flip_HV", transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=5, translate=(0, 0.1)),
            transforms.ColorJitter(contrast=0.5)
        ])),
        ("Gaussian_Noise", write_noisy_image),
        ("Gaussian_blurr", gaussian_blur)
    ]

    for transform in transformations:
        output_name = f'./{img}_r_{transform[0]}_{num}.jpg'
        t = transform[1](open_image)
        t.save(output_name)

def write_noisy_image(img):
    image = numpy.array(img) / 255.0
    image = skimage.util.random_noise(image, mode='gaussian', mean=0.01) * 255
    image = Image.fromarray(image.astype(np.uint8), 'RGB')

    return image

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(65,65),random.randrange(2, 4))
    image = Image.fromarray(image_blur)
    return image

def main():
    # Control loop for data creation
    # Control loop for data creation
    for fish in tqdm(categories):
        fish_dir = root_dir + fish[0] + "/"
        if len(fish[1]) > 1:
            for i in range(1, len(fish[1])):
                os.remove(fish_dir + fish[1][i])

        random_image = fish_dir + fish[1][0]

        for i in range(10):
            create_image(random_image, i)

if __name__ == '__main__':
    main()

