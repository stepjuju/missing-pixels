import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os

def pixelate_image(image, percent_pixels=0.1):
    """
    Randomly corrupts a percentage of the pixels of the image.

    Args:
    image (numpy.ndarray): A grayscale image of shape (H, W) or (1, H, W).
    percent_pixels (float): The percentage of total pixels to remove.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The corrupted image.
        - numpy.ndarray: The mask indicating which pixels were removed (0 for removed, 1 for kept).
    """
    # Ensure image is in the correct shape
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)  # Convert from (1, H, W) to (H, W)
    elif image.ndim != 2:
        raise ValueError("Image must be of shape (H, W) or (1, H, W)")

    # Create a mask with the same dimensions as the image
    mask = np.ones_like(image, dtype=np.uint8)

    # Calculate the number of pixels to remove
    total_pixels = image.shape[0] * image.shape[1]
    num_pixels_to_remove = int(total_pixels * percent_pixels)

    # Randomly select pixels to remove
    indices_to_remove = np.random.choice(total_pixels, num_pixels_to_remove, replace=False)

    # Convert linear indices to 2D indices
    indices_to_remove = np.unravel_index(indices_to_remove, image.shape)

    # Set selected pixels to zero in the image and update the mask
    image[indices_to_remove] = 0
    mask[indices_to_remove] = 0

    return  torch.from_numpy(image), torch.from_numpy(mask)

def load_image(image_path):
    """Load an image from the disk."""
    with Image.open(image_path) as img:
        return np.array(img)

def main():
    image_path = r'C:\Users\plani\MyProjects\missing-pixels\data\training_preprocessed\000'
    first_image_file = os.listdir(image_path)[0]
    full_image_path = os.path.join(image_path, first_image_file)

    # Load the image
    image = load_image(full_image_path)

    # Apply the pixelation function
    pixelated_image, mask = pixelate_image(image.copy(), percent_pixels=0.1)

    # Plotting the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(pixelated_image, cmap='gray')
    axes[1].set_title('Pixelated Image')
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Mask of Removed Pixels')
    axes[2].axis('off')

    plt.show()

if __name__ == '__main__':
    main()
