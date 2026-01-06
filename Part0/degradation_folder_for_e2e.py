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
    hr_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Step 2: Blur the image (Gaussian filter)
    blur1 = cv2.GaussianBlur(hr_resized, (15, 15), 0)
    
    # Step 3: Downsample to target LR size using bicubic interpolation
    lr_resized = cv2.resize(blur1, (144, 144), interpolation=cv2.INTER_CUBIC)

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

    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        lr_class_path = os.path.join(lr_dataset_path, class_name)

        # Create corresponding class folder in the LR directory
        os.makedirs(lr_class_path, exist_ok=True)

        for image_name in os.listdir(class_path):
            if image_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)

                if image is not None:
                    hr_image, degraded_image = apply_degradation(image)

                    hr_output_image_path = os.path.join(class_path, image_name)
                    if cv2.imwrite(hr_output_image_path, hr_image):
                        print(f"Saved degraded GT resized image: {hr_output_image_path}")

                    lr_output_image_path = os.path.join(lr_class_path, image_name)
                    
                    if cv2.imwrite(lr_output_image_path, degraded_image):
                        print(f"Saved degraded LR image: {lr_output_image_path}")
                    else:
                        print(f"Failed to save degraded LR image: {lr_output_image_path}")
                else:
                    print(f"Failed to load image at {image_path}")