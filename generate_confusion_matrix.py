import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import os
import glob
import sys

# Import from other project files
from train import build_model

# --- 1. Configuration ---
VAL_DIR    = "processed_data/val"
MODEL_DIR  = "models"
IMG_SIZE   = 224
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Helper Functions ---

def find_latest_model(model_dir):
    """Finds the most recently created model file in the directory."""
    list_of_files = glob.glob(os.path.join(model_dir, '*.pt'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_dataloader(data_dir, img_size, batch_size):
    """Loads the data for evaluation (without augmentation)."""
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def evaluate(model, dataloader, device):
    """Runs the model on the dataloader and returns all labels and predictions."""
    model.eval()
    all_preds = []
    all_labels = []

    print("Running evaluation to get predictions...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("Evaluation complete.")
    return all_labels, all_preds

def create_and_save_confusion_matrix(labels, preds, class_names, output_filename="confusion_matrix.png"):
    """Creates, displays, and saves a visualized confusion matrix."""
    cm = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.title('Confusion Matrix', fontsize=18)
    
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Confusion Matrix saved to '{output_filename}'")
    
    plt.show()

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # 1. Find the latest model automatically
    MODEL_PATH = find_latest_model(MODEL_DIR)
    if not MODEL_PATH:
        print(f"ERROR: No model found in directory '{MODEL_DIR}'. Please train a model first.")
        sys.exit(1)
    print(f"Using latest model: {MODEL_PATH}")

    # 2. Load class mapping
    MAPPING_PATH = os.path.join(MODEL_DIR, "class_mapping.json")
    if not os.path.exists(MAPPING_PATH):
        print(f"ERROR: Class mapping '{MAPPING_PATH}' not found. Make sure train.py has been run.")
        sys.exit(1)
        
    with open(MAPPING_PATH, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    CLASS_NAMES = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 3. Load model architecture and weights
    print("Loading model...")
    model = build_model(num_classes=len(CLASS_NAMES), device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 4. Load validation data
    print("Loading validation data...")
    val_loader = get_dataloader(VAL_DIR, IMG_SIZE, BATCH_SIZE)
    
    # 5. Get predictions and create the matrix
    labels, preds = evaluate(model, val_loader, DEVICE)
    create_and_save_confusion_matrix(labels, preds, CLASS_NAMES)
    print("\nScript finished successfully.")