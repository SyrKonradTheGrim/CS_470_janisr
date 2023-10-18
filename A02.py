# Program to perform convolution
# Written by Ryan Janis with segments written by Michael Reale
# Oct 2023 --- Python 3.10.11

import cv2
import numpy as np
import gradio as gr

def read_kernel_file(filepath):
    # Function to read the file for the kernel to get ready for the actual convolution.
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    with open(filepath, 'r') as file:
        line = file.readline() # Read the first line from the file
        
        tokens = line.split() # Split the line into tokens by spaces
        
        row_count, col_count = int(tokens[0]), int(tokens[1])
        kernel = np.zeros((row_count, col_count), dtype=np.float64)

        index = 2 #keeps track of the current position in the list of tokens read from the kernel file
        for i in range(row_count):
            for j in range(col_count):
                kernel[i, j] = float(tokens[index])
                index += 1

    return kernel

def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    # Function to apply the filter (Like in the name)
    # Written by Ryan Janis
    # Oct 2023 --- Python 3.10.11
    image = image.astype(np.float64)
    kernel = cv2.flip(kernel, -1)
    
    kernel_height, kernel_width = kernel.shape
    padding = (kernel_height // 2, kernel_width // 2) # // to force integer division
    
    padded_image = cv2.copyMakeBorder(image, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT, value=0)
    
    output = np.zeros(image.shape, dtype=np.float64) # Create an output array to store the filtered image
    
    for row in range(image.shape[0]): # Iterate through each possible center pixel
        for col in range(image.shape[1]):
            sub_image = padded_image[row : (row + kernel_height), col : (col + kernel_width)]
            filter_vals = sub_image * kernel
            value = np.sum(filter_vals)
            output[row, col] = value
    
    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    
    return output

def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    # Function that does something
    # Written by Michael Reale (I assume)
    # Sometime in spacetime in the universe --- Python 3.10.11
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img



def main():
    # The main that we all know and love
    # Written by Michael Reale (I assume)
    # Sometime in spacetime in the universe --- Python 3.10.11
    demo = gr.Interface(fn=filtering_callback,
                        inputs=["image",
                                "file",
                                gr.Number(value=0.125),
                                gr.Number(value=127)],
                        outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()