# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 12/13/2024
# Modified by : Eunji Ko
# Number      : 22100034
# Description:
#               - This code tests the SRGAN model and generates a restored high-resolution dataset.
#               - Modify the "# === Adjust" section to fit your dataset and environment.
#               - Input: Low-Resolution test dataset, Trained SRGAN model / Output: High-Resolution test dataset.
#               - The generated high-resolution images can later be used for image quality evaluation or classification tasks.

import argparse
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from model import Generator

parser = argparse.ArgumentParser(description='Test Multiple Images using SRGAN')
parser.add_argument('--upscale_factor', default=4, type=int, help='Super-resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='Use GPU or CPU')
parser.add_argument('--input_dir', type=str, default='data/test_data/LR', help='Directory for low-resolution test images')
parser.add_argument('--output_dir', type=str, default='data/test_data/HR', help='Directory for saving high-resolution images')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='Generator model filename')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
INPUT_DIR = opt.input_dir
OUTPUT_DIR = opt.output_dir
MODEL_NAME = opt.model_name

# Load SRGAN Generator model
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu')))

# === Adjust: List of classes
classes = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Convert images in each class folder to high resolution
for class_name in classes:
    class_input_dir = os.path.join(INPUT_DIR, class_name)
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(class_output_dir, exist_ok=True)
    
    for img_name in os.listdir(class_input_dir):
        img_path = os.path.join(class_input_dir, img_name)
        
        # Load and preprocess the image
        image = Image.open(img_path).convert('RGB')
        image = Variable(ToTensor()(image)).unsqueeze(0)
        
        if TEST_MODE:
            image = image.cuda()
        
        # Generate high-resolution image using the SRGAN model
        start = time.time()
        with torch.no_grad():
            out = model(image)
        elapsed = time.time() - start
        print(f'Processed {img_name} in {elapsed:.4f}s')
        
        # Save the high-resolution image
        out_img = ToPILImage()(out[0].cpu())
        out_img.save(os.path.join(class_output_dir, img_name))
