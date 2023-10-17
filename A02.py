import cv2
import numpy as np
import gradio as gr

def read_kernel_file(filepath):
    with open(filepath, 'r') as file:
        line = file.readline()
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError("Kernel file should contain at least row and column counts.")
        
        row_count, col_count = int(tokens[0]), int(tokens[1])
        kernel = np.zeros((row_count, col_count), dtype=np.float64)

        if len(tokens) != row_count * col_count + 2:
            raise ValueError(f"Expected {row_count * col_count} values in the kernel file, but found {len(tokens) - 2}.")

        idx = 2
        for i in range(row_count):
            for j in range(col_count):
                kernel[i, j] = float(tokens[idx])
                idx += 1

    return kernel

def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = cv2.flip(kernel, -1)
    
    kernel_height, kernel_width = kernel.shape
    padding = (kernel_height // 2, kernel_width // 2)
    
    padded_image = cv2.copyMakeBorder(image, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT, value=0)
    
    output = np.zeros(image.shape, dtype=np.float64)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            sub_image = padded_image[row : (row + kernel_height), col : (col + kernel_width)]
            filter_vals = sub_image * kernel
            value = np.sum(filter_vals)
            output[row, col] = value
    
    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    
    return output

def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img



def main():
    demo = gr.Interface(fn=filtering_callback,
                        inputs=["image",
                                "file",
                                gr.Number(value=0.125),
                                gr.Number(value=127)],
                        outputs=["image"])
    demo.launch()

# Later, at the bottom
if __name__ == "__main__":
    main()