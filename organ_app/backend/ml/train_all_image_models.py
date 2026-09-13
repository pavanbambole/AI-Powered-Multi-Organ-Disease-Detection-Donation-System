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
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATASET_BASE = os.path.join(BACKEND_DIR, 'data', 'dataset')
MODELS_DIR = os.path.join(BACKEND_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def get_transforms():
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
        ])
    }

def build_model_candidate(arch_name, num_classes=2):
    """
    Builds transfer learning model candidates with adapted classification heads.
    Supported: mobilenet_v2, resnet18, efficientnet_b0, densenet121
    """
    if arch_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze initial feature layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
    elif arch_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in list(model.features.parameters())[:-8]:
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
    elif arch_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in list(model.features.parameters())[:-8]:
            param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
    else:  # mobilenet_v2
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in list(model.features.parameters())[:-6]:
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
    return model

def train_candidate(arch_name, dataloaders, dataset_sizes, num_classes, epochs=12, lr=1e-3, device='cpu'):
    print(f"\n--- Training Candidate: {arch_name} ---")
    model = build_model_candidate(arch_name, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / dataset_sizes['train']
        train_acc = (running_corrects.double() / dataset_sizes['train']).item()

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['validation']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / dataset_sizes['validation']
        val_acc = (val_corrects.double() / dataset_sizes['validation']).item()
        scheduler.step(val_loss)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    print(f"Candidate {arch_name} -> Best Val Accuracy: {best_val_acc * 100:.1f}%, Best Val Loss: {best_val_loss:.4f}")
    return {
        'architecture': arch_name,
        'val_accuracy': round(best_val_acc * 100, 2),
        'val_loss': round(best_val_loss, 4),
        'best_weights': best_weights
    }

def evaluate_on_unseen_test(model, test_loader, class_names, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = round(float(accuracy_score(y_true, y_pred)) * 100, 2)
    prec = round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
    rec = round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
    f1 = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    per_class = {}
    for idx, cname in enumerate(class_names):
        c_true = (y_true == idx).astype(int)
        c_pred = (y_pred == idx).astype(int)
        per_class[cname] = {
            'precision': round(float(precision_score(c_true, c_pred, zero_division=0)) * 100, 2),
            'recall': round(float(recall_score(c_true, c_pred, zero_division=0)) * 100, 2),
            'f1_score': round(float(f1_score(c_true, c_pred, zero_division=0)) * 100, 2),
            'samples': int(np.sum(c_true))
        }

    return {
        'test_accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class_performance': per_class,
        'test_samples': len(y_true)
    }

def train_and_select_organ_model(organ, candidate_archs=['mobilenet_v2', 'resnet18', 'efficientnet_b0']):
    print(f"\n==================================================")
    print(f"  MODEL SELECTION & TRAINING: {organ.upper()}")
    print(f"==================================================")

    organ_dataset_dir = os.path.join(DATASET_BASE, organ)
    transforms_dict = get_transforms()

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(organ_dataset_dir, x), transforms_dict[x])
        for x in ['train', 'validation', 'test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=8, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'validation', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    class_to_idx = image_datasets['train'].class_to_idx

    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Compute device: {device}")

    # 1. Compare candidate models on validation split
    candidate_results = []
    for arch in candidate_archs:
        res = train_candidate(
            arch,
            dataloaders,
            dataset_sizes,
            num_classes=len(class_names),
            epochs=14,
            lr=8e-4,
            device=device
        )
        candidate_results.append(res)

    # 2. Select the champion model
    candidate_results.sort(key=lambda x: (x['val_accuracy'], -x['val_loss']), reverse=True)
    champion_res = candidate_results[0]
    champion_arch = champion_res['architecture']
    print(f"\n>>> Champion Model Selected for {organ.upper()}: {champion_arch} with {champion_res['val_accuracy']}% Val Accuracy")

    # 3. Evaluate Champion on Unseen Test Split
    champion_model = build_model_candidate(champion_arch, num_classes=len(class_names))
    champion_model.load_state_dict(champion_res['best_weights'])
    champion_model = champion_model.to(device)

    test_metrics = evaluate_on_unseen_test(champion_model, dataloaders['test'], class_names, device=device)
    print(f">>> Test Set Evaluation (Unseen Data):")
    print(f"    Test Accuracy: {test_metrics['test_accuracy']}%")
    print(f"    Precision:     {test_metrics['precision']}%")
    print(f"    Recall:        {test_metrics['recall']}%")
    print(f"    F1 Score:      {test_metrics['f1_score']}%")
    print(f"    Confusion Mat: {test_metrics['confusion_matrix']}")
    print(f"    Per-Class:     {test_metrics['per_class_performance']}")

    # 4. Save model checkpoints and metadata
    # Save directory: models/{organ}_image_model/
    organ_model_dir = os.path.join(MODELS_DIR, f"{organ}_image_model")
    os.makedirs(organ_model_dir, exist_ok=True)

    save_payload = {
        'organ': organ,
        'architecture': champion_arch,
        'model_state_dict': champion_res['best_weights'],
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'test_accuracy': test_metrics['test_accuracy'],
        'validation_accuracy': champion_res['val_accuracy'],
        'metrics': test_metrics,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }

    # Modular location
    model_modular_path = os.path.join(organ_model_dir, 'model.pt')
    torch.save(save_payload, model_modular_path)

    metrics_modular_path = os.path.join(organ_model_dir, 'metrics.json')
    with open(metrics_modular_path, 'w', encoding='utf-8') as f:
        json.dump({
            'organ': organ,
            'champion_model': champion_arch,
            'classes': class_names,
            'class_to_idx': class_to_idx,
            'validation_accuracy': champion_res['val_accuracy'],
            'test_accuracy': test_metrics['test_accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'confusion_matrix': test_metrics['confusion_matrix'],
            'per_class_performance': test_metrics['per_class_performance'],
            'test_samples': test_metrics['test_samples'],
            'evaluated_at': save_payload['trained_at']
        }, f, indent=2)

    comparison_path = os.path.join(organ_model_dir, 'model_comparison.json')
    comparison_summary = [
        {'architecture': c['architecture'], 'val_accuracy': c['val_accuracy'], 'val_loss': c['val_loss']}
        for c in candidate_results
    ]
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2)

    # Backwards-compatible root files
    root_model_path = os.path.join(MODELS_DIR, f"{organ}_image_model.pt")
    torch.save(save_payload, root_model_path)

    root_metrics_path = os.path.join(MODELS_DIR, f"{organ}_image_metrics.json")
    with open(root_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'organ': organ,
            'model_name': f"{champion_arch.upper()}-{organ.capitalize()}-Ultrasound",
            'architecture': champion_arch,
            'classes': class_names,
            'class_to_idx': class_to_idx,
            'validation_accuracy': champion_res['val_accuracy'],
            'test_accuracy': test_metrics['test_accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'confusion_matrix': test_metrics['confusion_matrix'],
            'per_class_performance': test_metrics['per_class_performance'],
            'test_samples': test_metrics['test_samples'],
            'trained_at': save_payload['trained_at']
        }, f, indent=2)

    print(f"Saved {organ.upper()} champion model to:")
    print(f"  - {model_modular_path}")
    print(f"  - {root_model_path}")

    return {
        'organ': organ,
        'champion_arch': champion_arch,
        'val_accuracy': champion_res['val_accuracy'],
        'test_accuracy': test_metrics['test_accuracy'],
        'metrics': test_metrics
    }

if __name__ == '__main__':
    print("Starting Automated AI Image Model Training & Selection Pipeline...")
    k_res = train_and_select_organ_model('kidney', ['mobilenet_v2', 'resnet18', 'efficientnet_b0'])
    l_res = train_and_select_organ_model('liver', ['mobilenet_v2', 'resnet18', 'efficientnet_b0'])

    print("\n==================================================")
    print("  FINAL MODEL SELECTION & PERFORMANCE SUMMARY")
    print("==================================================")
    print(f"Kidney Champion: {k_res['champion_arch']} -> Val Acc: {k_res['val_accuracy']}%, Test Acc: {k_res['test_accuracy']}%")
    print(f"Liver Champion:  {l_res['champion_arch']} -> Val Acc: {l_res['val_accuracy']}%, Test Acc: {l_res['test_accuracy']}%")
