import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def soft_mask(img, soft_mask_threshold=0.3, soft_mask_steepness=20):
    mask = 1.0 - img
    transition_point = soft_mask_threshold
    steepness = soft_mask_steepness
    mask = torch.sigmoid(steepness * (mask - transition_point))
    return mask

if __name__ == "__main__":
    # Generate a 1D array of "pixel" values ranging from 0 to 1 (simulating grayscale image pixel intensities)
    img_values = np.linspace(0, 1, 500)

    # Convert to torch tensor
    img_tensor = torch.tensor(img_values, dtype=torch.float32)

    # Apply the soft mask function
    mask_values = soft_mask(img_tensor).numpy()

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(img_values, mask_values)
    plt.title('Soft Mask Function')
    plt.xlabel('Input Pixel Value')
    plt.ylabel('Mask Value')
    plt.grid(True)
    plt.show()
