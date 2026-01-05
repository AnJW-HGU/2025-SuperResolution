import os
import shutil
import random

# === Adjust: Folder Paths
original_dataset_dir = 'dataset/originals' # Path to the originals images
reducing_dataset_dir = 'dataset/original' # Path to save reducing dataset

# === Adjust: Reducing Number
reducing_number = 400

def reduce_dataset():
    # Get class names from the original dataset
    class_names = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]

    reducing_images = []

    for class_name in class_names:
        # directory name + class name
        original_class_dir = os.path.join(original_dataset_dir, class_name)
        reducting_class_dir = os.path.join(reducing_dataset_dir, class_name)

        images = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]

        # Shuffe images
        random.shuffle(images)

        # original 수가 reducing 수보다 작은 경우
        slice_idx = min(len(images), reducing_number)

        # Slice images
        reducing_images_class = images[:slice_idx]

        os.makedirs(reducting_class_dir, exist_ok=True)
        for img in reducing_images_class:
            src = os.path.join(original_class_dir, img)
            dst = os.path.join(reducting_class_dir, img)
            shutil.copy2(src, dst)
            reducing_images.append(f"{class_name}/{img}")
    
if __name__ == "__main__":
    reduce_dataset()
<<<<<<< HEAD



=======
>>>>>>> 32a76eee4a7dbd72c1b8cdafb62fdbcb30902fbc
