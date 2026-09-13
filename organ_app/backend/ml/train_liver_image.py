import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data', 'dataset', 'liver'))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'liver_image_model.pt')
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, 'liver_image_metrics.json')

def get_data_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def build_model(num_classes=2):
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze base feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the top feature blocks for fine-tuning
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, num_classes)
    )
    return model

def train_and_evaluate(epochs=16, batch_size=16, lr=8e-4):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using compute device: {device}")

    transforms_dict = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), transforms_dict[x])
        for x in ['train', 'validation', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'validation', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    class_to_idx = image_datasets['train'].class_to_idx
    print(f"[INFO] Liver dataset sizes: {dataset_sizes}")
    print(f"[INFO] Class mapping: {class_names} -> {class_to_idx}")

    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.features[-4:].parameters(), 'lr': lr * 0.15},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_patience = 6
    patience_counter = 0

    print(f"\n{'='*50}\nStarting Liver Ultrasound Model Training\n{'='*50}")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 20)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize():10} Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

            if phase == 'validation':
                scheduler.step()
                if epoch_acc >= best_val_acc:
                    best_val_acc = epoch_acc
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f"  --> Checkpoint saved (New Best Val Acc: {best_val_acc * 100:.2f}%)")
                else:
                    patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\n[INFO] Early stopping triggered after {epoch + 1} epochs without improvement.")
            break

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training finished in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"[INFO] Best Validation Accuracy: {best_val_acc * 100:.2f}% (Loss: {best_val_loss:.4f})")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save model checkpoint
    torch.save({
        'model_state_dict': best_model_wts,
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'architecture': 'mobilenet_v2',
        'input_size': [3, 224, 224],
        'val_accuracy': float(best_val_acc),
        'val_loss': float(best_val_loss)
    }, MODEL_SAVE_PATH)
    print(f"[+] Saved model checkpoint to {MODEL_SAVE_PATH}")

    # =========================================================
    # RIGOROUS EVALUATION ON HELD-OUT TEST SPLIT
    # =========================================================
    print(f"\n{'='*50}\nEvaluating on Held-Out Test Set ({dataset_sizes['test']} unseen images)\n{'='*50}")
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"Test Set Metrics:")
    print(f"  • Real Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  • Precision:         {precision:.4f}")
    print(f"  • Recall:            {recall:.4f}")
    print(f"  • F1-Score:          {f1:.4f}")
    print(f"  • Confusion Matrix:  {cm}")

    metrics_payload = {
        'model_name': 'MobileNetV2-Liver-Ultrasound',
        'dataset': 'B-Mode Liver Ultrasound Histopathology & Fibrosis (xmidy/CNN_LIVER_FIBROSIS)',
        'classes': class_names,
        'class_to_idx': class_to_idx,
        'test_samples_count': dataset_sizes['test'],
        'test_accuracy': round(float(test_acc) * 100, 2),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': cm,
        'val_accuracy': round(float(best_val_acc) * 100, 2),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[+] Saved evaluation metrics to {METRICS_SAVE_PATH}")
    return metrics_payload

if __name__ == '__main__':
    train_and_evaluate(epochs=15, batch_size=8, lr=1e-3)
