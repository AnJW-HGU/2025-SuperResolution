# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 11/18/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code creates a low-resolution dataset to pair with a high-resolution dataset for Super Resolution tasks.
#               - The degradation method is based on the Real-ESRGAN paper.

import os
import cv2
import numpy as np

def apply_degradation(image):
    """
    Applies a series of degradation processes to the input image.
    Returns the resized high-resolution image and the degraded low-resolution image.
    """
    # === Adjust: Resize to match the dataset's target size
    # Step 1: Resize to HR target size
    hr_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)

    # Step 2: Blur the image (Gaussian filter)
    blur1 = cv2.GaussianBlur(hr_resized, (15, 15), 0)
    
    # Step 3: Downsample to target LR size using bicubic interpolation
    lr_resized = cv2.resize(blur1, (160, 160), interpolation=cv2.INTER_CUBIC)

    # Step 4: Add Gaussian noise
    noise = np.random.normal(0, 25, lr_resized.shape).astype(np.uint8)
    noisy_image = cv2.add(lr_resized, noise)
    
    # Step 5: Apply JPEG compression
    _, encoded_img = cv2.imencode('.jpg', noisy_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    compressed_image = cv2.imdecode(encoded_img, 1)
    
    # Step 6: Apply a second Gaussian blur
    blur2 = cv2.GaussianBlur(compressed_image, (5, 5), 0)
    
    return hr_resized, blur2

# === Adjust: Folder Paths
base_path = "dataset/gt"  # Path to source high-resolution images
lr_base_path = "dataset/lq"  # Path to save the degraded low-resolution images

# Iterate through train and test folders
for dataset in ["train", "test"]:
    dataset_path = os.path.join(base_path, dataset)
    lr_dataset_path = os.path.join(lr_base_path, dataset)

    # Create LR train/test directory if it doesn't exist
    os.makedirs(lr_dataset_path, exist_ok=True)

    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(('.jpeg', '.jpg', '.png')):
            image_path = os.path.join(dataset_path, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                hr_image, degraded_image = apply_degradation(image)

                hr_output_image_path = os.path.join(dataset_path, image_name)
                if cv2.imwrite(hr_output_image_path, hr_image):
                    print(f"Saved degraded GT resized image: {hr_output_image_path}")

                lr_output_image_path = os.path.join(lr_dataset_path, image_name)
                
                if cv2.imwrite(lr_output_image_path, degraded_image):
                    print(f"Saved degraded LR image: {lr_output_image_path}")
                else:
                    print(f"Failed to save degraded LR image: {lr_output_image_path}")
            else:
                print(f"Failed to load image at {image_path}")

    # # === Adjust: Class names based on the dataset
    # # Iterate through each class folder within train/test
    # for class_folder in ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]:
    #     class_path = os.path.join(dataset_path, class_folder)
    #     lr_class_path = os.path.join(lr_dataset_path, class_folder)

    #     # Create corresponding class folder in the LR directory
    #     os.makedirs(lr_class_path, exist_ok=True)

    #     for image_name in os.listdir(class_path):
    #         if image_name.lower().endswith(('.jpeg', '.jpg', '.png')):  # Process JPEG, JPG, PNG files
    #             image_path = os.path.join(class_path, image_name)
    #             image = cv2.imread(image_path)

    #             if image is not None:
    #                 hr_image, degraded_image = apply_degradation(image)

    #                 # Save degraded LR image
    #                 lr_output_image_path = os.path.join(lr_class_path, image_name)
                    
    #                 if cv2.imwrite(lr_output_image_path, degraded_image):
    #                     print(f"Saved degraded LR image: {lr_output_image_path}")
    #                 else:
    #                     print(f"Failed to save degraded LR image: {lr_output_image_path}")
    #             else:
    #                 print(f"Failed to load image at {image_path}")
