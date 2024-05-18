"""
Resize images for testing
"""

from PIL import Image
import os

def preprocess_images(input_folder, output_folder, target_size=(200, 250)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Open image
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            # Check if image is portrait (height > width)
            width, height = img.size
            if height > width:
                # Resize image to target size (keeping aspect ratio)
                img_resized = img.resize(target_size, Image.LANCZOS)

                # Save resized image to output folder
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)

if __name__ == "__main__":
    input_folder = "sample"
    output_folder = "resized_img"

    preprocess_images(input_folder, output_folder)
