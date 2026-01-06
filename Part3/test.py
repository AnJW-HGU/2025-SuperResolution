import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
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
num_epochs = 10
num_workers = 12
batch_size = 4 
learning_rate = 0.001

# === Adjust: Input Image Size 
transform_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# === Adjust: Dataset, Folder path
test_dataset = datasets.ImageFolder('dataset/lq/test', transform=transform_128)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Testing function (includes accuracy, precision, recall, and per-class metrics)
def test(model_g, model_cls, test_loader, class_names):
    model_g.eval()
    model_cls.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            g_outputs = model_g(images)
            cls_outputs = model_cls(g_outputs.data)
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

if __name__ == "__main__":

    test_model_g = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, 
        num_block=23, num_grow_ch=32, scale=4
    ).to(device)

    num_classes = len(test_dataset.classes)
    test_model_cls = models.resnet50(pretrained=False).to(device)
    test_model_cls.fc = nn.Linear(test_model_cls.fc.in_features, num_classes).to(device)

    def load_weights(model, path):
        checkpoint = torch.load(path, map_location=device)
        # BasicSR 저장 방식: 'params' 키 안에 가중치가 있음
        if 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded: {path}")


    load_weights(test_model_g, r"external\Real-ESRGAN\experiments\E2E_1_all_loss\models\net_g_latest.pth")
    load_weights(test_model_cls, r"external\Real-ESRGAN\experiments\E2E_1_all_loss\models\net_cls_latest.pth")

    # Get class names
    class_names = test_dataset.classes

    # === Adjust: Print 
    # Test the model and display results
    accuracy_e2e, precision_e2e, recall_e2e = test(test_model_g, test_model_cls, test_loader, class_names)
    print(f"\nOverall Accuracy for dataset: {accuracy_e2e * 100:.2f}%")
    print(f"Overall Precision for dataset: {precision_e2e * 100:.2f}%")
    print(f"Overall Recall for dataset: {recall_e2e * 100:.2f}%")