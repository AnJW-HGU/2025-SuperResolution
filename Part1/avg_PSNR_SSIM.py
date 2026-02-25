import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Paths ===
original_image_dir_path = "dataset/gt/test"          # 원본 이미지 폴더
low_res_image_dir_path = "dataset/SR/output/test"    # SR 결과 이미지 폴더

# 허용할 이미지 확장자
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

psnr_values = []
ssim_values = []
processed_count = 0

# === Iterate over GT images ===
for filename in os.listdir(original_image_dir_path):
    if not filename.lower().endswith(valid_exts):
        continue

    gt_path = os.path.join(original_image_dir_path, filename)
    sr_path = os.path.join(low_res_image_dir_path, filename)

    if not os.path.exists(sr_path):
        print(f"[WARNING] SR image not found: {filename}")
        continue

    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)

    if gt_img is None or sr_img is None:
        print(f"[WARNING] Failed to load image: {filename}")
        continue

    # Resize SR image to GT size if needed
    if gt_img.shape != sr_img.shape:
        sr_img = cv2.resize(
            sr_img,
            (gt_img.shape[1], gt_img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    # PSNR & SSIM
    psnr_value = psnr(gt_img, sr_img)
    ssim_value = ssim(gt_img, sr_img, channel_axis=2)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)
    processed_count += 1

# === Final Results ===
if processed_count > 0:
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print("\n====== Evaluation Result ======")
    print(f"Total images evaluated : {processed_count}")
    print(f"Average PSNR           : {avg_psnr:.4f}")
    print(f"Average SSIM           : {avg_ssim:.4f}")
else:
    print("No valid image pairs were processed.")
