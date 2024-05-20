import torch
import requests
import copy
import time
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from io import BytesIO
from torch import nn
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

class InitialChessClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(InitialChessClassifier, self).__init__()

        # Download pre-trained VGG16 model weights (without verification)
        vgg16_weights_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        response = requests.get(vgg16_weights_url, verify=False)
        weights_data = BytesIO(response.content)

        # Load pre-trained VGG16 model
        self.features = models.vgg16()
        self.features.load_state_dict(torch.load(weights_data))

        # Freeze pre-trained model parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Get final feature output size from pre-trained model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Assuming 3 channels, 224x224 image
            in_features = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4096),  # Input size based on VGG16
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ChessClassifierWithBatchNorm(nn.Module):
    def __init__(self, num_classes=6):
        super(ChessClassifierWithBatchNorm, self).__init__()

        # Download pre-trained VGG16 model weights (without verification)
        vgg16_weights_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        response = requests.get(vgg16_weights_url, verify=False)
        weights_data = BytesIO(response.content)

        # Load pre-trained VGG16 model
        self.features = models.vgg16()
        self.features.load_state_dict(torch.load(weights_data))

        # Freeze pre-trained model parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Get final feature output size from pre-trained model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            in_features = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4096),  # Input size based on VGG16
            nn.BatchNorm1d(4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def copy_files(file_paths, labels, target_dir):
    for file_path, label in zip(file_paths, labels):
        target_path = os.path.join(target_dir, label, os.path.basename(file_path))
        shutil.copy(file_path, target_path)

def train_model(model, criterion, optimizer, dataloaders, image_datasets, num_epochs=10, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    writer = SummaryWriter()  # for TensorBoard logging

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = None

    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if phase == 'train' and scheduler:
                scheduler.step()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    writer.close()  # Close the TensorBoard writer
    return model, train_losses, train_accuracies, val_losses, val_accuracies, best_epoch

def evaluate_model(model, dataloaders, class_folders):
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in dataloaders['test']:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=class_folders, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return report, conf_matrix

def save_checkpoint(model, dataloader, epoch, train_losses, train_acc, val_losses, val_acc, checkpoint_path):
    torch.save({
        'arch': 'vgg16',
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': dataloader['train'].dataset.class_to_idx,
        'epoch': epoch, 
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1], 
        'train_acc': train_acc[-1], 
        'val_acc': val_acc[-1],  
    }, checkpoint_path)

def plot_evaluation(class_folders, precision, recall, f1_score, img_name):
    # Plot precision, recall, and F1-score
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(class_folders))
    width = 0.2

    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1-Score')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1-Score per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_folders)
    ax.legend()

    plt.savefig(f'report/img/{img_name}.png')
    plt.show()

def plot_conf_matrix(class_folders, conf_matrix, img_name):
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_folders, yticklabels=class_folders)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'report/img/{img_name}.png')
    plt.show()