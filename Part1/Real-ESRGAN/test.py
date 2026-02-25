import yaml
import torch
from realesrgan.models.realesrgan_model import RealESRGANModel

# === Adjust: File Path
# Path to the configuration file
yml_path = 'external/Real-ESRGAN/experiments/2_Stage_64k/finetune_realesrgan_x4plus_pairdata.yml'

# Load settings from the YML file
with open(yml_path, 'r') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)

# Add 'dist' key
opt['dist'] = False
opt['is_train'] = False

# Create the model
model = RealESRGANModel(opt)

# === Adjust: File Path
# Path to the model weights
pth_path = 'external/Real-ESRGAN/experiments/2_Stage_64k/models/net_g_latest.pth'

# Load the model weights
checkpoint = torch.load(pth_path)
model.net_g.load_state_dict(checkpoint['params_ema'])  # Use the appropriate key

# Confirm the model is loaded
print("Model weights loaded successfully.")

import os
import torch
from torchvision import transforms
from PIL import Image

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.net_g.to(device)  # Transfer net_g to GPU
model.net_g.eval()  # Set net_g to evaluation mode

# Load and preprocess an image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()  # Convert image to tensor
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)

# Save an image
def save_image(tensor, output_path):
    tensor = tensor.squeeze(0).cpu().detach()  # Remove batch dimension
    img = transforms.ToPILImage()(tensor)  # Convert tensor to image
    img.save(output_path)

# Convert low-resolution images to high-resolution
def upscale_image(model, lr_image_path, output_path):
    lr_image = load_image(lr_image_path)
    with torch.no_grad():  # Disable gradient computation
        sr_image = model.net_g(lr_image)  # Generate high-resolution image
    save_image(sr_image, output_path)

# === Adjust: Folder Path
# Input and output folder paths
input_train_base_path = 'dataset/lq/train'  # Low-resolution image folder
input_test_base_path = 'dataset/lq/test'  # Low-resolution image folder
output_train_base_path = 'dataset/SR/output/train'  # High-resolution image output folder
output_test_base_path = 'dataset/SR/output/test'  # High-resolution image output folder

# === Adjust: Classes
# Class folder names
# classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'] # Case 1
# classes = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil'] # Case 2

# Process images for each class
# for class_name in classes:
#     input_class_path = os.path.join(input_base_path, class_name)  # Input folder path for the class
#     output_class_path = os.path.join(output_base_path, class_name)  # Output folder path for the class

# Create output folder if it doesn't exist 
os.makedirs(output_train_base_path, exist_ok=True)

# Process each image in the class folder
for image_name in os.listdir(input_train_base_path):
    lr_image_path = os.path.join(input_train_base_path, image_name)  # Path to the low-resolution image
    output_image_path = os.path.join(output_train_base_path, image_name)  # Path to save the high-resolution image

    # Upscale and save the image
    upscale_image(model, lr_image_path, output_image_path)

# Create output folder if it doesn't exist 
os.makedirs(output_test_base_path, exist_ok=True)

# Process each image in the class folder
for image_name in os.listdir(input_test_base_path):
    lr_image_path = os.path.join(input_test_base_path, image_name)  # Path to the low-resolution image
    output_image_path = os.path.join(output_test_base_path, image_name)  # Path to save the high-resolution image

    # Upscale and save the image
    upscale_image(model, lr_image_path, output_image_path)

print("All images have been processed and saved.")