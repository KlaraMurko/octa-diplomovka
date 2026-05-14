import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image


class OCTADataset(Dataset):
    def __init__(self, csv_path, data_all, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_all = data_all
        self.transform = transform
        self.classes = sorted(self.df['quality_label'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_all, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['quality_label']]
        if self.transform:
            image = self.transform(image)
        return image, label


class OCTAClassifier(nn.Module):
    def __init__(self, backbone, classifier):
        super(OCTAClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


def prepare_split(csv_path, data_all, split_dir, test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(csv_path, sep=None, engine='python')

    files_on_disk = set(os.listdir(data_all))
    df['exists'] = df['filename'].isin(files_on_disk)
    missing = df[~df['exists']]
    if len(missing) > 0:
        print(f"Upozornenie: {len(missing)} obrázkov sa nenašlo na disku, budú preskočené.")
    df = df[df['exists']].drop(columns=['exists'])

    print(f"Celkový počet obrázkov: {len(df)}")
    print(f"Rozdelenie tried:\n{df['quality_label'].value_counts().to_string()}\n")

    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size),
                                         random_state=random_state, stratify=df['quality_label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                       random_state=random_state, stratify=temp_df['quality_label'])

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(split_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(split_dir, 'test_split.csv'), index=False)

    print(f"CSV súbory so splitom boli uložené do: {split_dir}")


def create_model(model_name, num_classes, img_size, dropout_rate, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)

    if hasattr(model, 'num_features'):
        num_features = model.num_features
    elif hasattr(model, 'head'):
        num_features = model.head.in_features
    else:
        num_features = model(torch.randn(1, 3, img_size, img_size)).shape[1]

    total_params = len(list(model.parameters()))
    freeze_until = int(total_params * 0.7)
    for i, param in enumerate(model.parameters()):
        if i < freeze_until:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}  |  Features: {num_features}  |  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.8),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.6),
        nn.Linear(256, num_classes)
    )

    return OCTAClassifier(model, classifier).to(device)


def load_trained_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_name   = checkpoint['model_name']
    class_names  = checkpoint['class_names']
    img_size     = checkpoint['config']['img_size']
    dropout_rate = checkpoint['config']['dropout_rate']

    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    num_features = backbone.num_features

    classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.8),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.6),
        nn.Linear(256, len(class_names))
    )

    model = OCTAClassifier(backbone, classifier)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def create_data_loaders(split_dir, data_all, batch_size, img_size, num_workers):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = OCTADataset(os.path.join(split_dir, 'train_split.csv'), data_all, transform=train_transforms)
    val_dataset   = OCTADataset(os.path.join(split_dir, 'val_split.csv'),   data_all, transform=eval_transforms)
    test_dataset  = OCTADataset(os.path.join(split_dir, 'test_split.csv'),  data_all, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{running_loss/len(loader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, save_predictions=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if save_predictions:
                all_probs.extend(probabilities.cpu().numpy())

            pbar.set_postfix({'loss': f'{running_loss/len(loader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    if save_predictions:
        return running_loss / len(loader), 100. * correct / total, all_preds, all_labels, all_probs
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels


def evaluate_on_test_set(model, test_loader, criterion, class_names, device, output_dir):
    print(f"\n{'='*70}")
    print("Vyhodnotenie testovacej množiny")
    print(f"{'='*70}\n")

    _, test_acc, test_preds, test_labels, test_probs = validate(model, test_loader, criterion, device, save_predictions=True)

    report = classification_report(test_labels, test_preds, target_names=class_names, digits=3)
    print(report)

    with open(os.path.join(output_dir, 'test_classification_report.txt'), 'w') as f:
        f.write(f"Test Set Accuracy: {test_acc:.2f}%\n")
        f.write("="*60 + "\n\n")
        f.write(report)

    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.title('Confusion Matrix - Test Set', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    test_summary = {
        'test_accuracy': float(test_acc),
        'per_class_metrics': {
            cn: {
                'precision': report_dict[cn]['precision'],
                'recall': report_dict[cn]['recall'],
                'f1-score': report_dict[cn]['f1-score'],
                'support': report_dict[cn]['support']
            } for cn in class_names
        },
        'confusion_matrix': cm.tolist()
    }
    with open(os.path.join(output_dir, 'test_summary.json'), 'w') as f:
        json.dump(test_summary, f, indent=4)

    print(f"Presnosť na testovacej množine: {test_acc:.2f}%")
    return test_acc


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(val_losses,   label='Val Loss',   linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.title('Training and Validation Loss', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', linewidth=2, marker='o', markersize=4)
    plt.plot(val_accs,   label='Val Acc',   linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.title('Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_results(output_dir, model_name, iteration, class_names,
                            preds, labels, probs, train_losses, train_accs,
                            val_losses, val_accs, best_val_acc, config_dict):
    report = classification_report(labels, preds, target_names=class_names, digits=3)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model: {model_name} (Iteration {iteration})\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write("="*60 + "\n\n")
        f.write(report)

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_training_history(train_losses, train_accs, val_losses, val_accs,
                          os.path.join(output_dir, 'training_history.png'))

    if probs is not None:
        df_data = {
            'true_label':      [class_names[l] for l in labels],
            'predicted_label': [class_names[p] for p in preds],
            'correct':         [labels[i] == preds[i] for i in range(len(labels))]
        }
        for i, cn in enumerate(class_names):
            df_data[f'prob_{cn}'] = [p[i] for p in probs]
        pd.DataFrame(df_data).to_csv(os.path.join(output_dir, 'validation_predictions.csv'), index=False)

    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    summary = {
        'model_name': model_name,
        'iteration': iteration,
        'best_val_accuracy': float(best_val_acc),
        'final_train_acc': float(train_accs[-1]),
        'final_val_acc': float(val_accs[-1]),
        'epochs_trained': len(train_losses),
        'per_class_metrics': {
            cn: {
                'precision': report_dict[cn]['precision'],
                'recall': report_dict[cn]['recall'],
                'f1-score': report_dict[cn]['f1-score'],
                'support': report_dict[cn]['support']
            } for cn in class_names
        },
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    config_to_save = config_dict.copy()
    config_to_save['device'] = str(config_dict['device'])
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)

    print(f"Výsledky boli uložené do: {output_dir}")


def train_and_evaluate(config, data_all, csv_path, split_dir):
    if not os.path.exists(os.path.join(split_dir, 'train_split.csv')):
        print("Pripravuje sa rozdelenie dát...")
        prepare_split(csv_path, data_all, split_dir)
    else:
        print("Rozdelenie dát už existuje, preskakuje sa príprava.")

    output_dir = os.path.join('results', f"{config['model_name']}_{config['iteration']}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training: {config['model_name']} (Iteration {config['iteration']})")
    print(f"{'='*70}")

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        split_dir, data_all, config['batch_size'], config['img_size'], config['num_workers']
    )

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    class_counts = torch.zeros(len(class_names))
    for _, label in train_dataset:
        class_counts[label] += 1
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(config['device'])

    print(f"Distribúcia tried: {class_counts.numpy().astype(int)}")

    model = create_model(config['model_name'], len(class_names), config['img_size'],
                         config['dropout_rate'], config['device'])

    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 70)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, config['device'])
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.7f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_name': config['model_name'],
                'iteration': config['iteration'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': config
            }, os.path.join(output_dir, 'model.pth'))
            print(f"Saved best model (val acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    training_time = time.time() - start_time

    checkpoint = torch.load(os.path.join(output_dir, 'model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, final_preds, final_labels, final_probs = validate(
        model, val_loader, criterion, config['device'], save_predictions=True)

    save_evaluation_results(
        output_dir, config['model_name'], config['iteration'], class_names,
        final_preds, final_labels, final_probs, train_losses, train_accs,
        val_losses, val_accs, best_val_acc, config)

    test_acc = evaluate_on_test_set(model, test_loader, criterion, class_names, config['device'], output_dir)


    print(f"\n{'=' * 70}")
    print("Tréning dokončený.")
    print(f"Model: {config['model_name']} (Iterácia {config['iteration']})")
    print(f"Najlepšia validačná presnosť: {best_val_acc:.2f}%")
    print(f"Testovacia presnosť:            {test_acc:.2f}%")
    print(f"Čas tréningu: {training_time / 60:.1f} minút")
    print(f"Výsledky uložené do: {output_dir}")
    print(f"{'=' * 70}\n")

    return output_dir, best_val_acc, test_acc


def classify_all_images(model_path, image_folder, master_csv, target_datasets, output_file, device):
    model, class_names, checkpoint = load_trained_model(model_path, device)
    img_size = checkpoint['config']['img_size']

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    master = pd.read_csv(master_csv, sep=None, engine='python')
    master['filename'] = master['image_path'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
    df_filtered = master[master['dataset'].isin(target_datasets)].copy()

    print(f"Obrázky na klasifikáciu: {len(df_filtered)}")

    results = []
    with torch.no_grad():
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc='Classifying'):
            img_path = os.path.join(image_folder, row['filename'])
            if not os.path.exists(img_path):
                continue
            try:
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                results.append({
                    'filename': row['filename'],
                    'dataset': row['dataset'],
                    'predicted_class': class_names[pred.item()],
                    'confidence': conf.item()
                })
            except Exception:
                continue

    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False)

    print(f"\nCelkom klasifikovaných: {len(df_results)}")
    print(f"\nDistribúcia podľa datasetu a triedy:")
    print(df_results.groupby(['dataset', 'predicted_class']).size().unstack(fill_value=0).to_string())

    ungradable_count = (df_results['predicted_class'] == 'ungradable').sum()
    usable_count = len(df_results) - ungradable_count

    print(f"\nCelkom obrázkov:      {len(df_results)}")
    print(f"Nepoužiteľné:        {ungradable_count} ({100 * ungradable_count / len(df_results):.1f}%)")
    print(f"Použiteľné (zvyšné):   {usable_count} ({100 * usable_count / len(df_results):.1f}%)")

    return df_results