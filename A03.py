# Program written to detect white blood cells from a dataset
# Written by Ryan Janis
# Nov 2023 --- Python 3.10.11

import numpy as np
import cv2
import skimage

def find_WBC(image):
    # Function find White Blood Cells
    # Written by Ryan Janis
    # Nov 2023 --- Python 3.10.11
    
    # Step 1
    segments = skimage.segmentation.slic(image, start_label=0)
    cnt = len(np.unique(segments))
    
    # Step 2
    group_means = np.zeros((cnt,3), dtype = "float32")
    for specific_group in range(cnt):
        mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
        mask_image = np.expand_dims(mask_image, axis = -1)
        group_means[specific_group] = cv2.mean(image, mask = mask_image)[0:3]
        
    # Step 3
    k = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, bestLabels, centers = cv2.kmeans(group_means, k, None, criteria, 11, cv2.KMEANS_RANDOM_CENTERS) 
    
    # Step 4
    target_color = np.array([255, 0, 0], dtype = "float32")
    distances = np.linalg.norm(centers - target_color, axis=1)
    closest_group_idx = np.argmin(distances)
    
    # Step 5
    new_centers = np.zeros((k, 3), dtype="float32")
    new_centers[closest_group_idx] = [255, 255, 255]
    
    # Step 6
    new_centers = new_centers.astype("uint8")
    colors_per_clump = new_centers[bestLabels.flatten()]
    
    # Step 7
    cell_mask = colors_per_clump[segments]
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    # Step 8
    retval, labels = cv2.connectedComponents(cell_mask)
    
    # Step 9
    bounding_boxes = []
    for i in range(1, retval):
        coords = np.where(labels == i)
        if len(coords[0]) > 0:
            ymin, xmin, ymax, xmax = min(coords[0]), min(coords[1]), max(coords[0]), max(coords[1])
            bounding_boxes.append((ymin, xmin, ymax, xmax))

    return bounding_boxes