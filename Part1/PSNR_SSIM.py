# Class       : 2024-2 Mechatronics Integration Project
# Created     : 12/13/2024
# Modified by : Eunji Ko
# Number      : 22100034
# Description:
#               - This code outputs the PSNR and SSIM, which are Image Quality Measurements.
#               - It compares the similarity between the original high-resolution and the restored high-resolution image.
#               - Modify the "# === Adjust" section to fit your dataset and environment.

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Adjust: file path
# File paths for the original high-resolution image and the downsampled image
original_image_path = "GT_E_0187.jpeg"  # Path to the original high-resolution image
low_res_image_path = "SRGAN_0187.jpeg"  # Path to the restored high-resolution image

# Lists to store PSNR and SSIM values
psnr_values = []
ssim_values = []

# Read the images
original_img = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
low_res_img = cv2.imread(low_res_image_path, cv2.IMREAD_COLOR)

# Check if the images are loaded properly
if original_img is None:
    print(f"Failed to load the original image file: {original_image_path}")
else:
    print(f"Successfully loaded the original image file: {original_image_path}")

if low_res_img is None:
    print(f"Failed to load the low-resolution image file: {low_res_image_path}")
else:
    print(f"Successfully loaded the low-resolution image file: {low_res_image_path}")

# Resize the low-resolution image to match the original image dimensions
if original_img is not None and low_res_img is not None:
    low_res_img_resized = cv2.resize(low_res_img, (original_img.shape[1], original_img.shape[0]))

    # Calculate PSNR and SSIM
    psnr_value = psnr(original_img, low_res_img_resized)
    ssim_value = ssim(original_img, low_res_img_resized, win_size=3, channel_axis=2)

    # Add the PSNR and SSIM values to the respective lists
    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    # Calculate the average PSNR and SSIM
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)

    # Print the results
    print(f'PSNR between the low-resolution and the original image: {average_psnr:.4f}')
    print(f'SSIM between the low-resolution and the original image: {average_ssim:.4f}')
