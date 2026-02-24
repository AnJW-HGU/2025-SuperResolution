import os
import cv2
import numpy as np
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Paths ===
original_image_root_path = "dataset/gt/test"
low_res_image_root_path = "dataset/SR/output/test"
meta_info_path = "dataset/meta_info/gt_test_images.txt"

# === Storage ===
psnr_values = []
ssim_values = []

class_psnr = defaultdict(list)
class_ssim = defaultdict(list)

processed_count = 0

# === Read meta info ===
with open(meta_info_path, "r") as f:
    image_list = [line.strip() for line in f if line.strip()]

# === Evaluation loop ===
for rel_path in image_list:
    # rel_path: EOSINOPHIL/_10_2634.jpeg
    class_name, image_name = rel_path.split("/")

    gt_path = os.path.join(original_image_root_path, class_name, image_name)
    sr_path = os.path.join(low_res_image_root_path, class_name, image_name)

    if not os.path.exists(gt_path) or not os.path.exists(sr_path):
        print(f"[WARNING] Missing file: {rel_path}")
        continue

    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)

    if gt_img is None or sr_img is None:
        print(f"[WARNING] Failed to load image: {rel_path}")
        continue

    # Resize SR → GT
    if gt_img.shape != sr_img.shape:
        sr_img = cv2.resize(
            sr_img,
            (gt_img.shape[1], gt_img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    # Metrics
    psnr_value = psnr(gt_img, sr_img)
    ssim_value = ssim(gt_img, sr_img, channel_axis=2)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    class_psnr[class_name].append(psnr_value)
    class_ssim[class_name].append(ssim_value)

    processed_count += 1

# === Final Results ===
print("\n========== Overall Evaluation ==========")
print(f"Total images evaluated : {processed_count}")
print(f"Average PSNR           : {np.mean(psnr_values):.4f}")
print(f"Average SSIM           : {np.mean(ssim_values):.4f}")

print("\n========== Per-Class Evaluation ==========")
for cls in sorted(class_psnr.keys()):
    avg_psnr = np.mean(class_psnr[cls])
    avg_ssim = np.mean(class_ssim[cls])
    print(f"[{cls}]  PSNR: {avg_psnr:.4f},  SSIM: {avg_ssim:.4f},  Count: {len(class_psnr[cls])}")
