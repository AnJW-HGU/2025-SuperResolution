# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 10/10/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code makes the LR images with HR dataset for training Real-ESRGAN for test.
#               - The degradation method is based on the Real-ESRGAN paper.

import os
import cv2
import numpy as np

def apply_degradation(image_path):
    # Step 1: Load the HR image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return None

    # Resize HR to 2048x2048
    hr_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)

    # Step 2: Blur (Gaussian filter)
    blur1 = cv2.GaussianBlur(hr_resized, (15, 15), 0)

    # Step 3: Resize (downsample to 512x512 using bicubic)
    lr_resized = cv2.resize(blur1, (160, 160), interpolation=cv2.INTER_CUBIC)

    # Step 4: Add Gaussian noise
    noise = np.random.normal(0, 25, lr_resized.shape).astype(np.uint8)
    noisy_image = cv2.add(lr_resized, noise)

    # Step 5: JPEG compression
    result, encoded_img = cv2.imencode('.jpg', noisy_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    compressed_image = cv2.imdecode(encoded_img, 1)

    # Step 6: Second order degradation (blur again)
    blur2 = cv2.GaussianBlur(compressed_image, (5, 5), 0)

    return hr_resized, blur2

# Define the path to the HR image
image_path = "test/HR/test.png"  # Replace with the actual HR image path
hr_image, degraded_image = apply_degradation(image_path)

if hr_image is not None and degraded_image is not None:
    # Save HR 2048x2048 image
    hr_output_image_path = os.path.join(os.getcwd(), 'HR_image.png')  # Save in the same folder as the script
    cv2.imwrite(hr_output_image_path, hr_image)
    print(f"Saved HR image: {hr_output_image_path}")

    # Save degraded LR 512x512 image
    lr_output_image_path = os.path.join(os.getcwd(), 'LR_image.png')  # Save in the same folder as the script
    cv2.imwrite(lr_output_image_path, degraded_image)
    print(f"Saved degraded LR image: {lr_output_image_path}")
else:
    print("Failed to process the image.")
