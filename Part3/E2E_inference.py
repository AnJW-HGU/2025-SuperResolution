# inference.py
# import sys
# import os

# sys.path.append(os.path.join(os.getcwd(), 'Real-ESRGAN'))

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from pair_dataset import PairedDataset
import random
import numpy as np
from SRClassifier import SRClassifier
from basicsr.archs.rrdbnet_arch import RRDBNet
from sklearn.metrics import precision_score, recall_score

# === Adjust: GPU number
# GPU configuration (uses GPU set via CUDA_VISIBLE_DEVICES)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# === Adjust
# Hyperparameters
num_epochs = 20
num_workers = 8
batch_size = 4
learning_rate = 0.001

# === Adjust: Input Image Size 
# Image transformation settings (576x576 resolution)
transform_576 = transforms.Compose([
    transforms.Resize((576, 576)),  # Resize to 576x576
    transforms.ToTensor(),
])

transform_256 = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 576x576
    transforms.ToTensor(),
])

transform_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform_64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# === Adjust: Dataset, Folder path
# Load training and testing datasets (Real-ESRGAN dataset)
print("Loading Real-ESRGAN datasets...")
# train_dataset = datasets.ImageFolder('dataset/original', transform=transform_128)
train_dataset = PairedDataset(
    lr_dir='dataset/CLS_lq/train',
    hr_dir='dataset/gt/train',
    transform_lr=transform_64,
    transform_hr=transform_256
)
test_dataset = datasets.ImageFolder('dataset/CLS_lq/test', transform=transform_64)

# class 별로 랜덤으로 특정 % 씩 데이터 가져오기
# targets = np.array(train_dataset.targets)
# indices = []

# for c in range(len(train_dataset.classes)):
#     class_indices = np.where(targets == c) [0]
#     sampl_size = max(1, int(len(class_indices) * 0.2))
#     sampled = random.sample(list(class_indices), sampl_size)
#     indices.extend(sampled)
# small_train_dataset = Subset(train_dataset, indices)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
smaill_train_loader = DataLoader(train_dataset, num_workers=num_workers, 
                                 batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, num_workers=num_workers, 
                         batch_size=batch_size, shuffle=False)

# Generator 불러오기
generator = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
# state_dict = ckpt['params'] if 'params' in ckpt else ckpt['state_dict']
generator.load_state_dict(torch.load('external\Real-ESRGAN\experiments/finetune_RealESRGANx4plus_5k_pairdata_archived_20251019_191430\models/net_g_5000.pth', map_location=device), strict=False)
generator.eval()

# Classifier 불러오기
classifier = models.resnet50(pretrained=True)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(train_dataset.classes))
# classifier.load_state_dict(torch.load('classifier.pth', map_location=device))
# classifier.eval()

# E2E 모델 생성
model_e2e = SRClassifier(generator, classifier).to(device)
model_e2e.eval()

# Loss function and optimizer
criterion_sr = nn.L1Loss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_e2e.parameters(), lr=learning_rate)

# Training function

def train(model, optimizer, train_loader):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (lr_images, hr_images, labels) in enumerate(train_loader):
            lr_images, hr_images, labels = lr_images.to(device), hr_images.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward
            pred, sr_images = model(lr_images)
            sr_loss = criterion_sr(sr_images, hr_images)
            cls_loss = criterion_cls(pred, labels)
            
            total = sr_loss * 0.1 + cls_loss * 1.0


            # backward
            total.backward()
            optimizer.step()

            total_loss += total.item()

            # 주기적으로 메모리 해제
            del lr_images, hr_images, labels, pred, sr_images, sr_loss, cls_loss, total
            # torch.mps.empty_cache()

        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss/len(train_loader):.4f}")

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
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
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
    precision_per_class = precision_score(all_labels, all_preds, labels=list(range(len(class_names))), average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, labels=list(range(len(class_names))), average=None, zero_division=0)

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
    # print("Starting training phase...")
    # train(model_e2e, optimizer, smaill_train_loader)
    # print("Training completed. Starting testing phase...")

    # torch.save(model_e2e.state_dict(), "models/E2E_model_state_dict_5000_02_20.pth")
    # torch.save(model_e2e, "models/E2E_model_5000_02_20.pth")

    # test_model = torch.load("models/E2E_model_5000_02_20.pth").to(device)
    model_e2e.load_state_dict(torch.load("models/E2E_model_state_dict_5000_02_20.pth"))
    # test_model = models.resnet50(pretrained=True)
    # test_model.fc = nn.Linear(test_model.fc.in_features, len(train_dataset.classes))
    # Get class names
    class_names = train_dataset.classes

    # === Adjust: Print 
    # Test the model and display results
    accuracy_RE, precision_RE, recall_RE = test(model_e2e, test_loader, class_names)
    print(f"\nOverall Accuracy for Real-ESRGAN dataset: {accuracy_RE * 100:.2f}%")
    print(f"Overall Precision for Real-ESRGAN dataset: {precision_RE * 100:.2f}%")
    print(f"Overall Recall for Real-ESRGAN dataset: {recall_RE * 100:.2f}%")