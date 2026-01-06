# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 11/18/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code generates train and test datasets based on text files.

import os
import shutil

def create_dataset_from_txt(txt_train, txt_test, original_dir, train_dataset_dir, test_dataset_dir):
    original_train_dir = os.path.join(original_dir, 'train')
    original_test_dir = os.path.join(original_dir, 'test')

    # Read train and test image paths from text files
    with open(txt_train, 'r') as f:
        train_images = [line.strip() for line in f.readlines()]
    with open(txt_test, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]

    # Create train dataset
    for img_path in train_images:
        img_basename = os.path.basename(img_path)
        src = os.path.join(original_train_dir, img_basename)
        dst = os.path.join(train_dataset_dir, img_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # Create test dataset
    for img_path in test_images:
        img_basename = os.path.basename(img_path)
        src = os.path.join(original_test_dir, img_basename)
        dst = os.path.join(test_dataset_dir, img_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def create_test_dataset_from_txt(txt_test, original_dir, test_dataset_dir):
    # Read train and test image paths from text files
    with open(txt_test, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]

    # Create test dataset
    for img_path in test_images:
        img_basename = os.path.basename(img_path)
        src = os.path.join(original_dir, img_basename)
        dst = os.path.join(test_dataset_dir, img_path)

        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

# === Adjust: Folder Paths
# Paths for the dataset and output directories
original_dir = 'dataset/SR/output'  # Path to original images
train_dataset_dir = 'dataset/CLS/train'  # Path to save train dataset
test_dataset_dir = 'dataset/CLS/test'  # Path to save test dataset

# Generate train and test datasets
# create_dataset_from_txt('../CS_dataset/CS_classification_dataset/CS_classification_data_info/train_images.txt',
#                                '../CS_dataset/CS_classification_dataset/CS_classification_data_info/test_images.txt',
#                                original_dir, train_dataset_dir, test_dataset_dir)
create_dataset_from_txt('dataset/meta_info/gt_train_images.txt', 'dataset/meta_info/gt_test_images.txt',
                               original_dir, train_dataset_dir, test_dataset_dir)