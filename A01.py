import numpy as np
import gradio as gr
import cv2 

def create_unnormalized_hist(image):
    hist = np.zeros(256, dtype=np.float32)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            hist[pixel_value] += 1.0

    return hist

def normalize_hist(hist):
    return hist / np.sum(hist)

def create_cdf(nhist):
    cdf = np.cumsum(nhist)
    return cdf.astype(np.float32)

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    # Calculate the unnormalized histogram
    unnormalized_histogram = create_unnormalized_hist(image)
    
    # Normalize the histogram
    normalized_histogram = normalize_hist(unnormalized_histogram)
    
    # Create the cumulative distribution function (CDF)
    cdf = create_cdf(normalized_histogram)

    if do_stretching: 
        # Perform histogram stretching on the CDF
        cdfMax = cdf[-1] - cdf[0]
        cdf = (cdf - np.min(cdf)) / cdfMax
    cdf = cdf * 255

    # Convert the CDF to a 1D numpy array of uint8
    int_transform = np.copy(cdf)
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]
    
    return int_transform

def do_histogram_equalize(image, do_stretching):
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
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback,
                        inputs=["image", "checkbox"],
                        outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()
