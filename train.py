import os
import json
import shutil
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys

# ——————————————————————————————
# 1) Configuration
# ——————————————————————————————
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = os.path.join("processed_data", "train")
VAL_DIR   = os.path.join("processed_data", "val")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if data directories exist
if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
    print(f"ERROR: Training ('{TRAIN_DIR}') or validation ('{VAL_DIR}') directory not found.")
    print("Please run the 'prepare_data.py' script first.")
    sys.exit(1)

# ——————————————————————————————
# 2) DataLoader Function
# ——————————————————————————————

def get_dataloaders(batch_size, img_size):
    
    # Define the transformations for the images
    # We will use two different transformations:
    # 1) One for training with augmentation (e.g., random flips, rotations)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5), # Flip 50% of the images
        transforms.RandomRotation(10),           # Rotate by up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slightly change brightness/contrast
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalization is important
    ])
    
    # 2) One for validation without augmentation (just resizing and normalization)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds, val_ds
# ——————————————————————————————
# 3) Model Building
# ——————————————————————————————
def build_model(num_classes, device):
    model = models.resnet18(weights="DEFAULT")
    in_features = model.fc.in_features
    
    # Replace the last layer (the "fully connected" layer).
    # NOTE: Statically, model.fc is type-hinted as nn.Linear. We are replacing it
    # with an nn.Sequential block, which is valid at runtime but flags a type error.
    # We use `# type: ignore` to tell the type checker that this is intentional.
    classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model.fc = classifier  # type: ignore
    return model.to(device)

# ——————————————————————————————
# 3.5) Feature Extraction Function
# ——————————————————————————————
def extract_features(model, dataloader, target_layer_name, device):
    """
    Extracts features from a specified intermediate layer of a model.
    Uses a forward hook to capture the layer's output.
    """
    features_list = []
    labels_list = []
    
    # This hook function will be called when the target layer executes its forward pass
    def get_features_hook(module, input, output):
        # The output is a tensor; we flatten it, detach from the graph, and move to CPU
        features_list.append(output.detach().view(output.size(0), -1).cpu().numpy())

    # Find the target layer by name and register the hook
    target_layer = dict(model.named_modules()).get(target_layer_name)
    if target_layer is None:
        raise ValueError(f"Layer '{target_layer_name}' not found in model.")
    
    handle = target_layer.register_forward_hook(get_features_hook)

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            model(inputs)  # Forward pass to trigger the hook
            labels_list.append(labels.cpu().numpy())

    handle.remove()  # Important: remove the hook after use

    # Concatenate features and labels from all batches
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

# ——————————————————————————————
# 4) Training Routine
# ——————————————————————————————
def train(model, train_loader, val_loader, epochs, device, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_acc = 0.0
    best_model_path = "" # Path to the best model

    for epoch in range(1, epochs + 1):
        
        # — Training —
        
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects / len(train_loader.dataset)
        print(f"[Train] Epoch {epoch}/{epochs}  "
              f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2%}")

        # — Validation —
        model.eval() # Switch model to evaluation mode (important for Dropout etc.)
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad(): # Don't calculate gradients, we are not learning here
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += torch.sum(preds == labels).item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc  = val_corrects / len(val_loader.dataset)
        print(f"[ Val ] Epoch {epoch}/{epochs}  "
              f"Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.2%}")

        # — Log metrics to TensorBoard —
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        # — Early Stopping & Saving the best model —
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Create a unique name for the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(MODEL_DIR, f"best_model_acc_{best_val_acc*100:.2f}%_{timestamp}.pt")
            
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New best model found! Acc: {best_val_acc:.2%}. Saved at: {best_model_path}\n")
        else:
            # Newline for better readability when there is no improvement
            print() 

    print("\n--- Training complete ---")
    if best_model_path:
        print(f"Best model was saved with an accuracy of {best_val_acc:.2%}.")
        # Create a stable path for the best model for easy access by other scripts
        stable_best_model_path = os.path.join(MODEL_DIR, "best_model.pt")
        shutil.copy(best_model_path, stable_best_model_path)
        print(f"Copied best model to '{stable_best_model_path}' for easy access.")
        
        # Load the state of the best model to return it (optional, but good practice)
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("No model was saved as no improvement was achieved.")
        
    return model

# ——————————————————————————————
# 5) Main Script
# ——————————————————————————————
def run_feature_selection_demo():
    """Demonstrates multi-stage feature selection."""
    print("--- Running Multi-Stage Feature Selection Demonstration ---")
    print(f"Using device: {DEVICE}\n")

    # 1. Load data (no augmentation needed for feature extraction)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    
    if not train_ds or not val_ds:
        print("ERROR: No images found. Please run 'prepare_all_data.py'.", file=sys.stderr)
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load a pre-trained ResNet18 model
    model = models.resnet18(weights="DEFAULT").to(DEVICE)

    # 3. Define stages (layers) for feature extraction
    # 'layer3': Mid-level features
    # 'avgpool': High-level features just before the final classification layer
    feature_stages = ['layer3', 'avgpool']
    print(f"Will extract features from stages: {feature_stages}\n")

    # Suppress convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # 4. Loop through stages, extract features, and train a simple classifier
    for stage in feature_stages:
        print(f"--- Evaluating STAGE: '{stage}' ---")
        
        print("Extracting training features...")
        X_train, y_train = extract_features(model, train_loader, stage, DEVICE)
        print("Extracting validation features...")
        X_val, y_val = extract_features(model, val_loader, stage, DEVICE)
        print(f"Feature vector shape for this stage: {X_train.shape}")

        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)
        accuracy = classifier.score(X_val, y_val)
        print(f"--> Validation Accuracy using features from '{stage}': {accuracy:.2%}\n")

    print("--- Demonstration finished ---")
    print("Compare the validation accuracies to see which 'stage' provides more discriminative features.")

if __name__ == "__main__":
    run_feature_selection_demo()
