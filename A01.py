# Program to create an unnormalizzed histogram, normalize it, and then transform it
# Written by Ryan Janis with segments written by Michael Reale
# Oct 2023 --- Python 3.10.11

import numpy as np
import gradio as gr
import cv2 

def create_unnormalized_hist(image):
    # Function to create an unnormalized histogram
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    hist = np.zeros(256, dtype=np.float32)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            hist[pixel_value] += 1.0

    return hist

def normalize_hist(hist):
    # Function to normalize the histogram
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    return hist / np.sum(hist)

def create_cdf(nhist):
    # Function to create the CDF
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    cdf = np.cumsum(nhist)
    return cdf.astype(np.float32)

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    # Function to create the transformation of stretching
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    unnormalized_histogram = create_unnormalized_hist(image)
    
    normalized_histogram = normalize_hist(unnormalized_histogram)
    
    cdf = create_cdf(normalized_histogram)

    if do_stretching: 
        # Perform histogram stretching on the CDF
        cdfMax = cdf[-1] - cdf[0]
        cdf = (cdf - np.min(cdf)) / cdfMax
    cdf = cdf * 255

    # Convert the CDF
    int_transform = np.copy(cdf)
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]
    
    return int_transform

def do_histogram_equalize(image, do_stretching):
    # Function to equalize the histogram
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    output = image.copy()
    int_transform = get_hist_equalize_transform(image, do_stretching)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            new_value = int_transform[pixel_value]
            output[i, j] = new_value

    return output

def intensity_callback(input_img, do_stretching):
    # Function that does something
    # Written by Michael Reale (I assume)
    # Sometime in spacetime in the universe --- Python 3.10.11
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    # The main that we all know and love
    # Written by Michael Reale (I assume)
    # Sometime in spacetime in the universe --- Python 3.10.11
    demo = gr.Interface(fn=intensity_callback,
                        inputs=["image", "checkbox"],
                        outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()
