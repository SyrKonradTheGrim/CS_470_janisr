# Program written to extract LBP features from an image set
# Written by Ryan Janis
# Nov 2023 --- Python 3.10.11

import cv2
import numpy as np
from General_A04 import *

def getOneLBPLabel(subimage, label_type):
    # Function to get the LBP for one subimage
    # Written by Ryan Janis
    # Nov 2023 --- Python 3.10.11
    center = subimage[1, 1]
    values = [subimage[0, 0], subimage[0, 1], subimage[0, 2],
              subimage[1, 2], subimage[2, 2], subimage[2, 1],
              subimage[2, 0], subimage[1, 0]]

    binary_values = [1 if val > center else 0 for val in values]

    # Count transitions from 0 to 1 and 1 to 0
    transitions = sum(1 for i in range(8) if binary_values[i] != binary_values[(i + 1) % 8])

    if transitions <= 2:
        # Count the number of 1s in binary_values
        ones_count = sum(binary_values)
        return ones_count
    else:
        return 9

def getLBPImage(image, label_type):
    # Function to generate and return the uniform LBP label image
    # Written by Ryan Janis
    # Nov 2023 --- Python 3.10.11
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0)
    lbp_image = np.zeros_like(image)

    for i in range(1, image.shape[0] + 1):
        for j in range(1, image.shape[1] + 1):
            subimage = padded_image[i-1:i+2, j-1:j+2]
            lbp_image[i-1, j-1] = getOneLBPLabel(subimage, None)

    return lbp_image

def getOneRegionLBPFeatures(subImage, label_type):
    # Function to return the LBP histogram
    # Written by Ryan Janis
    # Nov 2023 --- Python 3.10.11
    histogram, _ = np.histogram(subImage, bins=np.arange(11), density=True)

    # If the histogram size is less than 10, pad with zeros
    if len(histogram) < 10:
        histogram = np.pad(histogram, (0, 10 - len(histogram)))

    # Normalize the histogram
    normalized_histogram = histogram.astype(float) / np.sum(histogram)

    return normalized_histogram

def getLBPFeatures(featureImage, regionSideCnt, label_type):
    # Function to get the LBP features
    # Written by Ryan Janis
    # Nov 2023 --- Python 3.10.11
    subregion_width = featureImage.shape[1] // regionSideCnt
    subregion_height = featureImage.shape[0] // regionSideCnt
    all_hists = []

    for i in range(regionSideCnt):
        for j in range(regionSideCnt):
            start_row = i * subregion_height
            end_row = (i + 1) * subregion_height
            start_col = j * subregion_width
            end_col = (j + 1) * subregion_width

            subimage = featureImage[start_row:end_row, start_col:end_col]
            hist = getOneRegionLBPFeatures(subimage, None)
            all_hists.append(hist)

    all_hists = np.array(all_hists)
    all_hists = np.reshape(all_hists, (all_hists.shape[0] * all_hists.shape[1],))

    return all_hists