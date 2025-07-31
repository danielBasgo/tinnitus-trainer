
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import json
import os
import glob
import sys

# ——————————————————————————————
# 1) Configuration
# ——————————————————————————————
MODEL_DIR       = "models"
VAL_DIR         = "processed_data/val"
IMG_SIZE        = 224
BATCH_SIZE      = 32
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILENAME = "confusion_matrix.png" # Filename for the saved plot

def find_latest_model(model_dir):
    """
    Finds the best model. Prefers 'best_model.pt', otherwise falls back
    to the most recently created model file in the directory.
    """
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        return best_model_path
        
    list_of_files = glob.glob(os.path.join(model_dir, '*.pt'))
    return max(list_of_files, key=os.path.getctime) if list_of_files else None

# ——————————————————————————————
# 2) Model and Data Setup
# ——————————————————————————————
from train import build_model

def get_dataloader(data_dir, img_size, batch_size):
    """Loads the data for evaluation (WITHOUT Augmentation)."""
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

# ——————————————————————————————
# 3) Evaluation Function
# ——————————————————————————————
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    print("Starting evaluation on the validation set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("Evaluation complete.")
    return all_labels, all_preds

def generate_and_save_report(labels, preds, class_names, output_filename):
    """
    Generates, prints, and saves a full evaluation report including a
    classification report and a confusion matrix plot.
    """
    # --- Classification Report ---
    print("\n--- Classification Report ---")
    report = classification_report(labels, preds, target_names=class_names)
    print(report)
    
    # --- Confusion Matrix ---
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.title('Confusion Matrix', fontsize=18)
    
    # Save the plot
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Confusion Matrix plot saved to '{output_filename}'")
    
    # Display the plot
    plt.show()

# ——————————————————————————————
# 4) Main Script
# ——————————————————————————————
def main():
    """Main function to run the evaluation pipeline."""
    # 1. Find the latest model automatically
    model_path = find_latest_model(MODEL_DIR)
    if not model_path:
        print(f"ERROR: No model found in directory '{MODEL_DIR}'. Please train a model first.")
        sys.exit(1)
    print(f"Using latest model: {os.path.basename(model_path)}")

    # 2. Load class mapping
    mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: Class mapping '{mapping_path}' not found. Make sure train.py has been run.")
        sys.exit(1)
        
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 3. Load model architecture and weights
    print("Loading model...")
    model = build_model(num_classes=len(class_names), device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # 4. Load validation data
    print("Loading validation data...")
    val_loader = get_dataloader(VAL_DIR, IMG_SIZE, BATCH_SIZE)
    
    # 5. Get predictions
    labels, preds = evaluate(model, val_loader, DEVICE)
    
    # 6. Generate and save all reports
    generate_and_save_report(labels, preds, class_names, OUTPUT_FILENAME)
    
    print("\n--- Evaluation script finished successfully. ---")

if __name__ == "__main__":
    main()