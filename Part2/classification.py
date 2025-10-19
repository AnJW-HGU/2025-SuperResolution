# Class       : 2024-2 Mechatronics Integration Project 
# Created     : 12/13/2024
# Author      : Eunji Ko
# Number      : 22100034
# Description:
#               - This code trains and tests a 'ResNet50 + Adaptive Pooling' classification model.
#               - It used the Real-ESRGAN High Resolution Image Dataset.
#               - You can modify "# === Adjust" Part as your dataset and environment.
#               - Input: Train & Test Image Folders / Output: Accuracy, Recall, Precision

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score

# === Adjust: GPU number
# GPU configuration (uses GPU set via CUDA_VISIBLE_DEVICES)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Adjust
# Hyperparameters
num_epochs = 20
num_workers = 12
batch_size = 8  
learning_rate = 0.001

# === Adjust: Input Image Size 
# Image transformation settings (576x576 resolution)
transform_576 = transforms.Compose([
    transforms.Resize((576, 576)),  # Resize to 576x576
    transforms.ToTensor(),
])

transform_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# === Adjust: Dataset, Folder path
# Load training and testing datasets (Real-ESRGAN dataset)
print("Loading Real-ESRGAN datasets...")
train_dataset_Real_ESRGAN = datasets.ImageFolder('dataset/original', transform=transform_128)
# test_dataset_Real_ESRGAN = datasets.ImageFolder('dataset/SR/test', transform=transform_128)
test_dataset_Real_ESRGAN = datasets.ImageFolder('dataset/original', transform=transform_128)

targets = np.array(train_dataset_Real_ESRGAN.targets)
indices = []

for c in range(len(train_dataset_Real_ESRGAN.classes)):
    class_indices = np.where(targets == c) [0]
    sample_size = max(1, int(len(class_indices) * 0.2))
    sampled = random.sample(list(class_indices), sample_size)
    indices.extend(sampled)
small_dataset_Real_ESRGAN = Subset(train_dataset_Real_ESRGAN, indices)

# train_loader_Real_ESRGAN = DataLoader(train_dataset_Real_ESRGAN, batch_size=batch_size, shuffle=True)
small_train_loader_Real_ESRGAN = DataLoader(train_dataset_Real_ESRGAN, 
                                            num_workers=num_workers, batch_size=batch_size, 
                                            shuffle=True)
test_loader_Real_ESRGAN = DataLoader(test_dataset_Real_ESRGAN, batch_size=batch_size, shuffle=False)

# === Adjust: Model Name
# Configure the ResNet model for the Real-ESRGAN dataset
model_Real_ESRGAN = models.resnet50(pretrained=False)
model_Real_ESRGAN.fc = nn.Linear(model_Real_ESRGAN.fc.in_features, len(train_dataset_Real_ESRGAN.classes))
model_Real_ESRGAN = model_Real_ESRGAN.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_Real_ESRGAN = optim.Adam(model_Real_ESRGAN.parameters(), lr=learning_rate)

# Training function
def train(model, optimizer, train_loader):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Record loss
            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Testing function (includes accuracy, precision, recall, and per-class metrics)
def test(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
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

if __name__ == "__main__":
    # Train and test using the Real-ESRGAN dataset
    print("Starting training phase...")
    train(model_Real_ESRGAN, optimizer_Real_ESRGAN, small_train_loader_Real_ESRGAN)
    print("Training completed. Starting testing phase...")

    torch.save(model_Real_ESRGAN.state_dict(), "models/original_state_dict_02_20.pth")
    torch.save(model_Real_ESRGAN, "models/original_02_20.pth")

    test_model = torch.load("models/original_02_20.pth").to(device)
    # test_model = models.resnet50(pretrained=False)
    # test_model.fc = nn.Linear(model_Real_ESRGAN.fc.in_features, len(train_dataset_Real_ESRGAN.classes))
    test_model.load_state_dict(torch.load("models/original_state_dict_02_20.pth"))
    # Get class names
    class_names_Real_ESRGAN = train_dataset_Real_ESRGAN.classes

    # === Adjust: Print 
    # Test the model and display results
    accuracy_Real_ESRGAN, precision_Real_ESRGAN, recall_Real_ESRGAN = test(test_model, test_loader_Real_ESRGAN, class_names_Real_ESRGAN)
    print(f"\nOverall Accuracy for Real-ESRGAN dataset: {accuracy_Real_ESRGAN * 100:.2f}%")
    print(f"Overall Precision for Real-ESRGAN dataset: {precision_Real_ESRGAN * 100:.2f}%")
    print(f"Overall Recall for Real-ESRGAN dataset: {recall_Real_ESRGAN * 100:.2f}%")
