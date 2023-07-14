#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
from tqdm import tqdm
from collections import Counter


# %%

import random
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel

#custom modules
from helper import rotateImage


# %%


# Get the variable from the command line argument
train_test = sys.argv[1]

current_folder_path = os.path.join('data/original_images', train_test)
rotated_folder_path = os.path.join('data/rotated_images', train_test)


# %%


# angles = [random.randint(40, 360) for _ in range(10)]


# %%

images_path = []
rotated_angle = []
import random
from collections import Counter

def rotated_data_creation(folder_path, angle_list=[0,90,180,270]):
    # Count the occurrences of each angle in the rotated_angle list
    angle_counts = Counter(rotated_angle)
    num_images = len(os.listdir(folder_path))
    min_angle_count = num_images // len(angle_list)

    for idx, imagename in enumerate(tqdm(os.listdir(folder_path))):
        if imagename.split(".")[-1] != "ipynb_checkpoints":
            img_path = os.path.join(folder_path, imagename)
            image = Image.open(img_path)

            # Check if any angle needs to be removed from angle_list
            for angle in angle_list:
                if angle_counts.get(angle, 0) > min_angle_count:
                    angle_list.remove(angle)

            # Select a random angle from the updated angle_list
            angle = random.choice(angle_list)

            # Rotate the image by the selected angle
            _, rotated_image = rotateImage(np.array(image), angle)

            # Image extension
            ext = img_path.split(".")[-1]

            # Save the rotated image
            output_path = os.path.join(rotated_folder_path, "{}_{}.{}".format(imagename.split(".")[0], angle, ext))
            # rotated_image.save(output_path)
            io.imsave(output_path, rotated_image)

            # Append the values
            images_path.append(output_path)
            rotated_angle.append(angle)

            # Increment the count of the selected angle
            angle_counts[angle] += 1
        else:
            continue

    return images_path, rotated_angle


# %%


path, angles = rotated_data_creation(current_folder_path)


# %%


labelled_data = pd.DataFrame({'path': path,
                             'angles': angles})
labelled_data.head()


# %%


labelled_data['angles'].value_counts()


# %%


labelled_data.to_csv('data/{}_data.csv'.format(train_test), index=False)


# %%




