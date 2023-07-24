#general libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

#PIL
from PIL import Image

#torch
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#skimage
from skimage.feature import hog
from skimage.filters import threshold_otsu, sobel
from skimage.io import imread, imshow
from skimage.transform import resize, rotate, hough_line, hough_line_peaks
from skimage import exposure, color
from skimage import io, transform, img_as_ubyte
from skimage.color import rgb2gray
from scipy.stats import mode

#more libraries
from skimage import io, color, transform, img_as_ubyte
import numpy as np
import os

#tqdm
from tqdm import tqdm

#warnings
import warnings
warnings.filterwarnings('ignore')

USE_OPENMP=1

#custom modules
from constants import label_mapping, device

##HOUGH transform functions
def binarizeImage(RGB_image):

    if len(RGB_image.shape) == 3 and RGB_image.shape[2] == 3:
        image = color.rgb2gray(RGB_image)
    else:
        image = RGB_image  # Assume grayscale image if not 3-channel RGB
    threshold = threshold_otsu(image)
    bina_image = image < threshold
    return bina_image

def findEdges(bina_image):
  
    image_edges = sobel(bina_image)

    # plt.imshow(bina_image, cmap='gray')
    # plt.axis('off')
    # plt.title('Binary Image Edges')
    # plt.savefig('binary_image.png')

    return image_edges

def findTiltAngle(image_edges):
  
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    # print(angles)
    # angle = np.rad2deg(mode(angles)[0][0])

    angles_mode = mode(angles)
    # print(angles_mode)
    angle = np.rad2deg(angles_mode.mode)

    # print(angle)
  
    if (angle < 0):
    
        r_angle = angle + 90
    
    else:
    
        r_angle = angle - 90

    # Plot Image and Lines    
    # fig, ax = plt.subplots()
  

    # ax.imshow(image_edges, cmap='gray')

    origin = np.array((0, image_edges.shape[1]))

#     for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

#         y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#         ax.plot(origin, (y0, y1), '-r')

#     ax.set_xlim(origin)
#     ax.set_ylim((image_edges.shape[0], 0))
#     ax.set_axis_off()
#     ax.set_title('Detected lines')

#     plt.savefig('hough_lines.jpg')

#     plt.show()

    return r_angle

def rotateImage(image, angle, img_name=None):
    # Convert RGB image to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = color.rgb2gray(image)
    else:
        image = image

    # Rotate the image
    rotated_image = transform.rotate(image, angle, resize=True)

    # Create a copy of the rotated image to preserve pixel values
    cropped_image = np.copy(rotated_image)

    # Find the bounding box of the rotated image
    nonzero_pixels = np.argwhere(rotated_image > 0)
    min_row, min_col = np.min(nonzero_pixels, axis=0)
    max_row, max_col = np.max(nonzero_pixels, axis=0)

    # Crop the rotated image to remove black borders
    cropped_image = cropped_image[min_row:max_row+1, min_col:max_col+1]

    # Convert the image to unsigned 8-bit integer format
    final_image = img_as_ubyte(cropped_image)

    # Save the resulting image
    # if img_name is None:
    # io.imsave('output_image.png', final_image)
    # else:
    #     os.makedirs('logs', exist_ok=True)
    #     io.imsave(os.path.join('logs', img_name), final_image)

    return cropped_image, final_image



def generalPipeline(img_path):

    img = img_path

    image = io.imread(img)
    bina_image = binarizeImage(image)
    image_edges = findEdges(bina_image)
    angle = findTiltAngle(image_edges)
    # if img_name is None:
    _, fixed_image = rotateImage(io.imread(img), angle)
    # else:
    #     fixed_image, _ = rotateImage(io.imread(img), angle, img_name)
    # print(fixed_image)
    return fixed_image

## Model training functions
def get_hog(path=None, pixels=None):
    
    if path is not None and pixels is not None:
        raise ValueError("Only one parameter (path or pixels) should be provided.")
    if path is None and pixels is None:
        raise ValueError("Either path or pixels should be provided.")
    if pixels is not None:
        image_array = pixels
        image_array = np.array(Image.fromarray(image_array).resize((256, 256)))
    else:
        image_pil = Image.open(path)
        image_pil = image_pil.convert("L")
        image_pil = image_pil.resize((256, 256))
        image_array = np.array(image_pil)

    # Define HOG parameters
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)

    # Compute HOG features
    features, hog_image = hog(image_array, orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              visualize=True, block_norm='L2-Hys')

    # Display the HOG image
    hog_image_pil = (hog_image * 255).astype(np.uint8)
    return hog_image_pil



class HOG_Dataset(Dataset):
    def __init__(self, df, threshold=None, transform=None):
        self.df = df
        self.path = df.path.values
        self.labels = df.angles.values
#         self.threshold = threshold

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file_path and label for index
        label = self.labels[idx]
        path = self.path[idx]

        #rotate to the nearest angle
        if label >= 0 and label < 90:
            # print("between 0 and 90")
            _, rotated_pixels = rotateImage(io.imread(path), 90-label)
            labels = 90

        if label >= 90 and label < 180:
            # print("between 90 and 180")
            _, rotated_pixels = rotateImage(io.imread(path), 180-label)
            labels = 180

        if label >= 180 and label < 270:
            # print("between 180 and 270")
            _, rotated_pixels = rotateImage(io.imread(path), 270-label)
            labels = 270

        if label >= 270 and label <= 360:
            # print("between 270 and 360")
            _, rotated_pixels = rotateImage(io.imread(path), 360-label)
            labels = 0

        #hog
        hog_pixels = get_hog(pixels=rotated_pixels)
        
        #transform
        features = self.transform(hog_pixels)
        label = label_mapping.get(labels)
        label = torch.tensor(label)
        
        label = label.to(device)
        features = features.to(device, dtype=torch.float)

        # features = torch.tensor(features, dtype=torch.float)

        return {'pixels': features,
               'label': label}

