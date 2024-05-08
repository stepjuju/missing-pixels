import os
from PIL import Image
from torchvision import transforms

def resize_image(img, size=(64, 64)):
    """Resize the image to the specified size."""
    return img.resize(size)

def is_grayscale(img):
    """Check if the image is grayscale."""
    if img.mode != 'RGB':
        return True
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True

def preprocess_images(input_dir, output_dir, target_size=(64, 64)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize transformation for grayscale conversion
    to_grayscale = transforms.Grayscale(num_output_channels=1)

    processed_count = 0
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, file)
                with Image.open(img_path) as img:
                    if img.size != target_size:
                        img = resize_image(img, target_size)
                    if not is_grayscale(img):
                        img = to_grayscale(img)

                    # Save the preprocessed image
                    output_path = os.path.join(output_dir, os.path.relpath(img_path, input_dir))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path.replace(os.path.splitext(output_path)[1], '.png'), 'PNG')  # Save as PNG
                    processed_count += 1

    print(f"Processed {processed_count} images.")

# Example usage
#input_path = r'C:\Users\'
#output_path = r'C:\Users\'
#preprocess_images(input_path, output_path)