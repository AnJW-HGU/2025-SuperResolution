# Class       : 2024-2 Mechatronics Integration Project
# Created     : 11/18/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code restores low-resolution datasets using Bicubic Interpolation.
#               - Modify the "# === Adjust" section to fit your dataset and environment.
#               - Input: Low-Resolution Datasets / Output: High-Resolution Dataset (restored via Bicubic).

import os
from PIL import Image
from pathlib import Path

# === Adjust: Folder Paths
# Paths for input (low-resolution) and output (high-resolution) datasets
base_input_folder = 'dataset/lq'  # Path to low-resolution images
base_output_folder = 'dataset/Bicudic/output'  # Path to save results
# Resize scale (e.g., 4x upscaling)
scale = 4

# Process both train and test folders
for dataset_type in ['train', 'test']:
    input_folder = os.path.join(base_input_folder, dataset_type)
    output_folder = os.path.join(base_output_folder, dataset_type)

    # Create folder structure and process images
    for root, dirs, files in os.walk(input_folder):
        # Get the class name from the current folder
        relative_path = Path(root).relative_to(input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Check for valid image extensions
                # Load the image
                img_path = os.path.join(root, file)
                img = Image.open(img_path)

                # Resize the image using Bicubic Interpolation
                bicubic_img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)

                # Save the resulting image
                output_path = os.path.join(output_dir, file)
                bicubic_img.save(output_path)

                print(f"Processed and saved: {output_path}")
