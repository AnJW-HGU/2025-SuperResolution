import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# === Adjust: GPU number
# GPU configuration (uses GPU set via CUDA_VISIBLE_DEVICES)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Adjust
# Hyperparameters
num_epochs = 20
num_workers = 16
batch_size = 8  
learning_rate = 0.001

# === Adjust: Input Image Size 
# Image transformation settings (256x256 resolution)
transform_HR = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to 256x256
    transforms.ToTensor(),
])

# === Adjust: Dataset, Folder path
# Load training and testing datasets (Real-ESRGAN dataset)
print("Loading Real-ESRGAN datasets...")
train_dataset_Real_ESRGAN = datasets.ImageFolder('dataset/CLS/train', transform=transform_HR)
test_dataset_Real_ESRGAN = datasets.ImageFolder('dataset/CLS/test', transform=transform_HR)

train_loader_Real_ESRGAN = DataLoader(train_dataset_Real_ESRGAN, batch_size=batch_size, shuffle=True)
test_loader_Real_ESRGAN = DataLoader(test_dataset_Real_ESRGAN, batch_size=batch_size, shuffle=False)

# === Adjust: Model Name
# Configure the ResNet model for the Real-ESRGAN dataset
model_Real_ESRGAN = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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

    cm = confusion_matrix(all_labels, all_preds)

    class_accuracy = []
    num_classes = len(class_names)
    total_samples = np.sum(cm)

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = total_samples - (TP + FP + FN)

        acc_i = (TP + TN) / total_samples
        class_accuracy.append(acc_i)


    # Calculate overall accuracy
    accuracy = np.mean(class_accuracy)

    # Calculate precision and recall per class
    precision_per_class = precision_score(all_labels, all_preds, labels=list(range(len(class_names))), average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, labels=list(range(len(class_names))), average=None, zero_division=0)

    # Print per-class metrics
    print("\nClass-wise Metrics:")
    for i, class_name in enumerate(class_names):
        # class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {class_name}:")
        print(f"    Accuracy: {class_accuracy[i]*100:.2f}%")
        print(f"    Precision: {precision_per_class[i] * 100:.2f}%")
        print(f"    Recall: {recall_per_class[i] * 100:.2f}%")

    # Calculate overall precision and recall
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall

if __name__ == "__main__":
    # Train and test using the Real-ESRGAN dataset
    print("Starting training phase...")
    train(model_Real_ESRGAN, optimizer_Real_ESRGAN, train_loader_Real_ESRGAN)
    print("Training completed. Starting testing phase...")

    # torch.save(model_Real_ESRGAN.state_dict(), "models/2_Stage_100k_state_dict.pth")
    # torch.save(model_Real_ESRGAN, "models/2_Stage_100k.pth")

    # test_model = torch.load("models/2_Stage_100k.pth", weights_only=False).to(device)
    # # test_model = models.resnet50(pretrained=False)
    # # test_model.fc = nn.Linear(model_Real_ESRGAN.fc.in_features, len(train_dataset_Real_ESRGAN.classes))
    # test_model.load_state_dict(torch.load("models/2_Stage_100k_state_dict.pth"))
    
    # Get class names
    class_names_Real_ESRGAN = train_dataset_Real_ESRGAN.classes

    # === Adjust: Print 
    # Test the model and display results
    accuracy_Real_ESRGAN, precision_Real_ESRGAN, recall_Real_ESRGAN = test(model_Real_ESRGAN, test_loader_Real_ESRGAN, class_names_Real_ESRGAN)
    print(f"\nOverall Accuracy for Real-ESRGAN dataset: {accuracy_Real_ESRGAN * 100:.2f}%")
    print(f"Overall Precision for Real-ESRGAN dataset: {precision_Real_ESRGAN * 100:.2f}%")
    print(f"Overall Recall for Real-ESRGAN dataset: {recall_Real_ESRGAN * 100:.2f}%")