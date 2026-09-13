import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
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
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data', 'dataset', 'kidney'))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))
KIDNEY_MODEL_DIR = os.path.join(MODELS_DIR, 'kidney_image_model')
os.makedirs(KIDNEY_MODEL_DIR, exist_ok=True)

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

def build_model_candidate(arch_name, num_classes=2):
    """
    Build transfer learning model candidates with tailored classification heads
    and fine-tuning on upper convolutional blocks.
    """
    if arch_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        # Freeze early layers
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
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
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
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
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
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
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
    print(f"\n==========================================")
    print(f" Training Candidate Model: {arch_name.upper()}")
    print(f"==========================================")
    model = build_model_candidate(arch_name, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_patience = 4
    patience_counter = 0

    for epoch in range(epochs):
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
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            if phase == 'validation':
                scheduler.step(epoch_loss)
                if epoch_acc > best_val_acc or (epoch_acc == best_val_acc and epoch_loss < best_val_loss):
                    best_val_acc = epoch_acc
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"  [Info] Early stopping at epoch {epoch + 1}")
            break

    print(f"  Best Val Accuracy ({arch_name}): {best_val_acc * 100:.2f}% (Loss: {best_val_loss:.4f})")

    return {
        'architecture': arch_name,
        'val_accuracy': round(best_val_acc * 100, 2),
        'val_loss': round(best_val_loss, 4),
        'best_weights': best_model_wts
    }

def evaluate_on_unseen_test(model, test_loader, class_names, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    test_acc = float(accuracy_score(all_labels, all_preds)) * 100
    test_prec = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)) * 100
    test_rec = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)) * 100
    test_f1 = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)) * 100
    cm = confusion_matrix(all_labels, all_preds).tolist()

    per_class = {}
    for i, cls in enumerate(class_names):
        cls_y = [1 if y == i else 0 for y in all_labels]
        cls_p = [1 if p == i else 0 for p in all_preds]
        per_class[cls] = {
            'precision': round(float(precision_score(cls_y, cls_p, zero_division=0)) * 100, 2),
            'recall': round(float(recall_score(cls_y, cls_p, zero_division=0)) * 100, 2),
            'f1_score': round(float(f1_score(cls_y, cls_p, zero_division=0)) * 100, 2),
            'samples': sum(cls_y)
        }

    return {
        'test_accuracy': round(test_acc, 2),
        'precision': round(test_prec, 2),
        'recall': round(test_rec, 2),
        'f1_score': round(test_f1, 2),
        'confusion_matrix': cm,
        'per_class_performance': per_class,
        'test_samples': len(all_labels)
    }

def train_and_select_kidney_model(candidate_archs=None):
    if candidate_archs is None:
        candidate_archs = ['resnet18', 'efficientnet_b0', 'mobilenet_v2', 'densenet121']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using compute device: {device}")

    transforms_dict = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), transforms_dict[x])
        for x in ['train', 'validation', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=8, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'validation', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    class_to_idx = image_datasets['train'].class_to_idx

    print(f"[INFO] Kidney Dataset Split Sizes: {dataset_sizes}")
    print(f"[INFO] Class Names & Mapping: {class_names} -> {class_to_idx}")

    candidate_results = []
    for arch in candidate_archs:
        res = train_candidate(arch, dataloaders, dataset_sizes, len(class_names), epochs=12, device=device)
        candidate_results.append(res)

    # Sort candidates by validation accuracy descending, then validation loss ascending
    candidate_results.sort(key=lambda c: (-c['val_accuracy'], c['val_loss']))
    champion_res = candidate_results[0]
    champion_arch = champion_res['architecture']

    print("\n==================================================")
    print(" MODEL SELECTION RESULTS (Validation Performance)")
    print("==================================================")
    for c in candidate_results:
        star = " [CHAMPION]" if c['architecture'] == champion_arch else ""
        print(f"  - {c['architecture'].upper():<16}: Val Acc = {c['val_accuracy']:.2f}%, Val Loss = {c['val_loss']:.4f}{star}")

    print(f"\n[INFO] Selected Champion Model Architecture: {champion_arch.upper()}")

    # Build champion and load best weights for held-out evaluation
    champion_model = build_model_candidate(champion_arch, num_classes=len(class_names))
    champion_model.load_state_dict(champion_res['best_weights'])
    champion_model = champion_model.to(device)

    print("\n[INFO] Running evaluation on completely UNSEEN HELD-OUT TEST SET...")
    test_metrics = evaluate_on_unseen_test(champion_model, dataloaders['test'], class_names, device=device)

    print("\n==================================================")
    print(" HELD-OUT TEST EVALUATION METRICS (GENUINE):")
    print(f"  Test Accuracy:    {test_metrics['test_accuracy']:.2f}%")
    print(f"  Precision:        {test_metrics['precision']:.2f}%")
    print(f"  Recall:           {test_metrics['recall']:.2f}%")
    print(f"  F1-Score:         {test_metrics['f1_score']:.2f}%")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")
    print(f"  Per-Class:        {json.dumps(test_metrics['per_class_performance'], indent=2)}")
    print("==================================================")

    trained_time = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

    save_payload = {
        'organ': 'kidney',
        'architecture': champion_arch,
        'model_state_dict': champion_res['best_weights'],
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'val_accuracy': champion_res['val_accuracy'],
        'test_accuracy': test_metrics['test_accuracy'],
        'metrics': test_metrics,
        'trained_at': trained_time
    }

    # 1. Save modular champion model
    modular_model_path = os.path.join(KIDNEY_MODEL_DIR, 'model.pt')
    torch.save(save_payload, modular_model_path)

    # 2. Save modular metrics
    modular_metrics_path = os.path.join(KIDNEY_MODEL_DIR, 'metrics.json')
    with open(modular_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'organ': 'kidney',
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
            'evaluated_at': trained_time
        }, f, indent=2)

    # 3. Save comparison summary
    comparison_path = os.path.join(KIDNEY_MODEL_DIR, 'model_comparison.json')
    comparison_summary = [
        {'architecture': c['architecture'], 'val_accuracy': c['val_accuracy'], 'val_loss': c['val_loss']}
        for c in candidate_results
    ]
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2)

    # 4. Save backwards-compatible root files
    root_model_path = os.path.join(MODELS_DIR, 'kidney_image_model.pt')
    torch.save(save_payload, root_model_path)

    root_metrics_path = os.path.join(MODELS_DIR, 'kidney_image_metrics.json')
    with open(root_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'organ': 'kidney',
            'model_name': f"{champion_arch.upper()}-Kidney-Ultrasound",
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
            'trained_at': trained_time
        }, f, indent=2)

    print(f"\n[SUCCESS] Saved Kidney Champion Model to:")
    print(f"  - {modular_model_path}")
    print(f"  - {modular_metrics_path}")
    print(f"  - {comparison_path}")
    print(f"  - {root_model_path}")
    print(f"  - {root_metrics_path}")

    return {
        'champion_arch': champion_arch,
        'val_accuracy': champion_res['val_accuracy'],
        'test_accuracy': test_metrics['test_accuracy'],
        'test_metrics': test_metrics
    }

if __name__ == '__main__':
    train_and_select_kidney_model()
