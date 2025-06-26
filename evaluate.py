
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import json
import os

# WICHTIG: Passen Sie diese Konfiguration an Ihr Projekt an!
# ——————————————————————————————
# 1) Konfiguration
# ——————————————————————————————
# FÜGEN SIE HIER DEN GENAUEN NAMEN IHRER BESTEN MODELLDATEI EIN!
MODEL_PATH = "models/best_model_acc_90.69%_20231027_183000.pt" # Beispiel, ersetzen Sie dies!
VAL_DIR    = "processed_data/val"
IMG_SIZE   = 224
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Laden der Klassenzuordnung
MAPPING_PATH = "models/class_mapping.json"
with open(MAPPING_PATH, 'r') as f:
    class_to_idx = json.load(f)
# Drehen Sie das Mapping um, um von Index zu Klassenname zu kommen
idx_to_class = {v: k for k, v in class_to_idx.items()}
CLASS_NAMES = [idx_to_class[i] for i in range(len(idx_to_class))]


# ——————————————————————————————
# 2) Modell- und Daten-Setup (aus train.py kopiert)
# ——————————————————————————————
# Wir brauchen die build_model Funktion, um die Architektur zu laden
from train import build_model # Wir importieren die Funktion aus Ihrer train.py Datei

def get_dataloader(data_dir, img_size, batch_size):
    """Lädt die Daten für die Evaluation (OHNE Augmentation)."""
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

# ——————————————————————————————
# 3) Evaluationsfunktion
# ——————————————————————————————
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    print("Starte Evaluation auf dem Validierungs-Set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("Evaluation abgeschlossen.")
    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, class_names):
    """Erstellt und zeigt eine visualisierte Confusion Matrix."""
    cm = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('Wirklicher Zustand (Actual)')
    plt.xlabel('Vorhersage (Predicted)')
    plt.title('Confusion Matrix', fontsize=16)
    plt.show()


# ——————————————————————————————
# 4) Hauptskript
# ——————————————————————————————
if __name__ == "__main__":
    # 1. Modell laden
    print(f"Lade Modell von: {MODEL_PATH}")
    model = build_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 2. Daten laden
    val_loader = get_dataloader(VAL_DIR, IMG_SIZE, BATCH_SIZE)
    
    # 3. Evaluation durchführen
    labels, preds = evaluate(model, val_loader, DEVICE)
    
    # 4. Ergebnisse anzeigen
    print("\n--- Classification Report ---")
    # Dieser Report zeigt Precision, Recall und F1-Score für jede Klasse
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    
    print("\n--- Confusion Matrix ---")
    plot_confusion_matrix(labels, preds, CLASS_NAMES)