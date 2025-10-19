# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 11/18/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code creates train and test folders for classification tasks with a user-defined ratio.
#               - It also generates text files to ensure consistency across datasets.

import os
import shutil
import random

# === Adjust: Folder Paths
# Paths for the original dataset and the train/test folders
original_dataset_dir = 'dataset/original'  # Path to the original images
train_dataset_dir = 'dataset/gt/train'  # Path to save train dataset
test_dataset_dir = 'dataset/gt/test'  # Path to save test dataset

# Paths for the text files to save image names
train_txt_file = 'dataset/meta_info/gt_train_images.txt'
test_txt_file = 'dataset/meta_info/gt_test_images.txt'

# === Adjust: Train-Test Split Ratio
train_ratio = 0.8  # Ratio for the training dataset

def create_dataset_split():
    # Get class names from the original dataset
    class_names = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]
    
    train_images = []
    test_images = []

    for class_name in class_names:
        class_dir = os.path.join(original_dataset_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train and test sets
        split_idx = int(len(images) * train_ratio)
        train_images_class = images[:split_idx]
        test_images_class = images[split_idx:]

        # # Copy train images and save filenames
        # train_class_dir = os.path.join(train_dataset_dir, class_name)
        # os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(train_dataset_dir, exist_ok=True)
        for img in train_images_class:
            src = os.path.join(class_dir, img)
            # dst = os.path.join(train_class_dir, img)
            dst = os.path.join(train_dataset_dir, img)
            shutil.copy2(src, dst)
            train_images.append(f"{class_name}/{img}")

        # # Copy test images and save filenames
        # test_class_dir = os.path.join(test_dataset_dir, class_name)
        # os.makedirs(test_class_dir, exist_ok=True)
        os.makedirs(test_dataset_dir, exist_ok=True)
        for img in test_images_class:
            src = os.path.join(class_dir, img)
        #     dst = os.path.join(test_class_dir, img)
            dst = os.path.join(test_dataset_dir, img)
            shutil.copy2(src, dst)
            test_images.append(f"{class_name}/{img}")

    # Write filenames to text files
    with open(train_txt_file, 'w') as f:
        for img in train_images:
            f.write(f"{img}\n")
    with open(test_txt_file, 'w') as f:
        for img in test_images:
            f.write(f"{img}\n")

if __name__ == "__main__":
    create_dataset_split()
