import os
import json
import shutil
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
import sys
from google.cloud import storage

# ——————————————————————————————
# 1) Configuration
# ——————————————————————————————
BATCH_SIZE = 32
EPOCHS     = 10
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_DATA_DIR = "/tmp/data"
LOCAL_MODEL_DIR = "/tmp/models"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# GCS Helper Functions
# ——————————————————————————————

def download_gcs_directory(bucket_name, source_directory, destination_directory):
    """Lädt einen kompletten Ordner von GCS herunter."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_directory)  # Liste alle Objekte im Ordner auf

    print(f"Downloading data from gs://{bucket_name}/{source_directory} to {destination_directory}...")
    for blob in blobs:
        # Erstelle die lokale Ordnerstruktur
        relative_path = os.path.relpath(blob.name, source_directory)
        local_path = os.path.join(destination_directory, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Lade die Datei herunter
        if not blob.name.endswith('/'):
            blob.download_to_filename(local_path)
    print("Download complete.")

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Lädt eine einzelne Datei nach GCS hoch."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")

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
    # Define the GCS bucket and data path
    GCS_BUCKET_NAME = "tinnitus-trainer-data"
    GCS_DATA_PATH = "processed_data"  # Path in GCS where the data is stored
    download_gcs_directory(GCS_BUCKET_NAME, GCS_DATA_PATH, LOCAL_DATA_DIR)

    local_train_dir = os.path.join(LOCAL_DATA_DIR, TRAIN_DIR)
    local_val_dir   = os.path.join(LOCAL_DATA_DIR, VAL_DIR)
    
    if not os.path.isdir(local_train_dir) or not os.path.isdir(local_val_dir):
        print(f"ERROR: Training ('{local_train_dir}') or validation ('{local_val_dir}') directory not found after download.")
        sys.exit(1)
    print(f"Using local directories:\n"
          f"  Training: {local_train_dir}\n"
          f"  Validation: {local_val_dir}")
    # If you want to upload the model to GCS after training, uncomment the following line:
    # upload_to_gcs(GCS_BUCKET_NAME, LOCAL_MODEL_DIR, "models/")
    
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
    
    train_ds = datasets.ImageFolder(local_train_dir, transform=train_transform)
    val_ds   = datasets.ImageFolder(local_val_dir,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds, val_ds
# ——————————————————————————————
# 3) Model Building
# ——————————————————————————————
def build_model(num_classes, device):
    model = models.resnet18(weights="DEFAULT")
    in_features = model.fc.in_features
    
    # Replace the last layer with a sequence
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),  # An additional layer
        nn.ReLU(),                    # Activation function
        nn.Dropout(0.5),              # DROPOUT: 50% of neurons are randomly deactivated
        nn.Linear(256, num_classes)   # The final output layer (dynamic)
    )
    return model.to(device)

# ——————————————————————————————
# 4) Training Routine
# ——————————————————————————————
def train(model, train_loader, val_loader, epochs, device):
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

        # Load the state of the best model to return it
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("No model was saved as no improvement was achieved.")
        
    return model

# ——————————————————————————————
# 5) Main Script
# ——————————————————————————————
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load data
    train_loader, val_loader, train_ds, val_ds = get_dataloaders(BATCH_SIZE, IMG_SIZE)
    
    # Ensure that data was found
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: No images found in the training or validation directories. Exiting script.")
        sys.exit(1)

# 2. Save class mapping for evaluate.py
class_to_idx = train_ds.class_to_idx
local_mapping_path = os.path.join(LOCAL_MODEL_DIR, "class_mapping.json")
with open(local_mapping_path, 'w') as f:
    json.dump(class_to_idx, f)
print(f"Class mapping saved locally at: {local_mapping_path}")

gcs_mapping_destination = f"{GCS_MODEL_OUTPUT_PATH}/class_mapping.json"
upload_to_gcs(GCS_BUCKET_NAME, local_mapping_path, gcs_mapping_destination)

num_classes = len(train_ds.classes)
print(f"Found classes: {num_classes} {train_ds.classes}")
model = build_model(num_classes=num_classes, device=DEVICE)

trained_model = train(model, train_loader, val_loader, EPOCHS, DEVICE)

    # Bereinige den temporären Datenordner am Ende
print(f"Cleaning up temporary data directory: {LOCAL_DATA_DIR}")
shutil.rmtree(LOCAL_DATA_DIR)
print("Cleanup complete.")

# 3. Build model (with the correct number of classes)
num_classes = len(train_ds.classes)
print(f"Found classes: {num_classes} {train_ds.classes}")
model = build_model(num_classes=num_classes, device=DEVICE)

# 4. Start training
trained_model = train(model, train_loader, val_loader, EPOCHS, DEVICE)