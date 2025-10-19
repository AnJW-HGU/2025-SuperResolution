# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 11/18/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code resizes the image dataset for experiments.
#               - Modify the "# === Adjust" section to suit your dataset and environment.
#               - Input: Train & Test Image Folders / Output: Resized Train & Test Image Folders

import os
from PIL import Image
from pathlib import Path

# === Adjust: Dataset and Folder Paths
input_train_folder = '../CS_dataset/CS_HR_woclass'
input_test_folder = '../CS_dataset/CS_HR/test'

output_train_folder = '../CS_dataset/CS_HR/resized_train'
output_test_folder = '../CS_dataset/CS_HR/resized_test'

# === Adjust: Target Image Size
target_size = (576, 576)

# Process images in the train folder
for root, dirs, files in os.walk(input_train_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Check for image file extensions
            # Load image
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            
            # Resize image
            img_resized = img.resize(target_size, Image.BICUBIC)
            
            # Create output directory
            relative_path = Path(root).relative_to(input_train_folder)
            output_dir = os.path.join(output_train_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save resized image
            output_path = os.path.join(output_dir, file)
            img_resized.save(output_path)
            print(f"Resized and saved: {output_path}")

# Process images in the test folder by class
for class_folder in os.listdir(input_test_folder):
    class_folder_path = os.path.join(input_test_folder, class_folder)
    if os.path.isdir(class_folder_path):
        for root, dirs, files in os.walk(class_folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Check for image file extensions
                    # Load image
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    
                    # Resize image
                    img_resized = img.resize(target_size, Image.BICUBIC)
                    
                    # Create output directory for the class
                    output_class_dir = os.path.join(output_test_folder, class_folder)
                    os.makedirs(output_class_dir, exist_ok=True)
                    
                    # Save resized image
                    output_path = os.path.join(output_class_dir, file)
                    img_resized.save(output_path)
                    print(f"Resized and saved: {output_path}")
