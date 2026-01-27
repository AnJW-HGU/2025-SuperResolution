import os
import torch
import yaml
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from realesrgan.models.realesrgan_model import RealESRGANModel
from sklearn.metrics import precision_score, recall_score

# === Adjust: GPU number
# GPU configuration (uses GPU set via CUDA_VISIBLE_DEVICES)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Adjust
# Hyperparameters
batch_size = 4 

yml_path = 'external/Real-ESRGAN\experiments/E2E_75k/finetune_realesrgan_x4plus_pairdata.yml'

with open(yml_path, 'r') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)

# Add 'dist' key
opt['dist'] = False
opt['is_train'] = False

model = RealESRGANModel(opt)

# === Adjust: File Path
# Path to the model weights
pth_g_path = 'external/Real-ESRGAN/experiments/E2E_75k/models/net_g_latest.pth'
pth_cls_path = 'external/Real-ESRGAN/experiments/E2E_75k/models/net_cls_latest.pth'

# Load the model weights
checkpoint_g = torch.load(pth_g_path)
checkpoint_cls = torch.load(pth_cls_path)
model.net_g.load_state_dict(checkpoint_g['params_ema']) 
# model.net_cls.load_state_dict(checkpoint_cls['params_ema']) 

# Confirm the model is loaded
print("Model weights loaded successfully.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.net_g.to(device)
model.net_cls.to(device)
model.net_g.eval()
model.net_cls.eval()


# === Adjust: Input Image Size 
transform_64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# === Adjust: Dataset, Folder path
test_dataset = datasets.ImageFolder('dataset/lq/test', transform=transform_64)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

def upscale_image(lr_image_path, sr_image_path):
    lr_image = load_image(lr_image_path)
    with torch.no_grad():  # Disable gradient computation
        sr_image = model.net_g(lr_image)  # Generate high-resolution image
    save_image(sr_image, sr_image_path)

# Testing function (includes accuracy, precision, recall, and per-class metrics)
def test(model, test_loader, class_names):
    model.net_g.eval()
    model.net_cls.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            sr_images = model.net_g(images)
            cls_outputs = model.net_cls(sr_images.data)
            _, predicted = torch.max(cls_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Record per-class predictions
            for label, prediction in zip(labels, predicted):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

            # Save all labels and predictions for precision/recall calculations
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate overall accuracy
    accuracy = correct / total

    # Calculate precision and recall per class
    precision_per_class = precision_score(all_labels, all_preds, labels=list(range(len(class_names))), average=None)
    recall_per_class = recall_score(all_labels, all_preds, labels=list(range(len(class_names))), average=None)

    # Print per-class metrics
    print("\nClass-wise Metrics:")
    for i, class_name in enumerate(class_names):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {class_name}:")
        print(f"    Accuracy: {class_acc * 100:.2f}%")
        print(f"    Precision: {precision_per_class[i] * 100:.2f}%")
        print(f"    Recall: {recall_per_class[i] * 100:.2f}%")

    # Calculate overall precision and recall
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall


# Get class names
class_names_dataset = test_dataset.classes

# === Adjust: Print 
# Test the model and display results
accuracy_e2e, precision_e2e, recall_e2e = test(model, test_loader, class_names_dataset)
print(f"\nOverall Accuracy for dataset: {accuracy_e2e * 100:.2f}%")
print(f"Overall Precision for dataset: {precision_e2e * 100:.2f}%")
print(f"Overall Recall for dataset: {recall_e2e * 100:.2f}%")


# === Adjust: Save sr images
input_test_base_path = 'dataset/lq/test'
output_test_base_path = 'dataset/SR/output/test'

os.makedirs(output_test_base_path, exist_ok=True)

# Process each image in the class folder
for class_name in os.listdir(input_test_base_path):
    lr_image_dir = os.path.join(input_test_base_path, class_name)
    sr_image_dir = os.path.join(output_test_base_path, class_name)

    os.makedirs(sr_image_dir, exist_ok=True)

    for image_name in os.listdir(lr_image_dir):
        lr_image_path = os.path.join(lr_image_dir, image_name)  # Path to the low-resolution image
        output_image_path = os.path.join(sr_image_dir, image_name)  # Path to save the high-resolution image

        # Upscale and save the image
        upscale_image(lr_image_path, output_image_path)

print("All images have been processed and saved.")