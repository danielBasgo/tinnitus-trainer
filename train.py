import os
from pyexpat import model
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime

# ——————————————————————————————
# 1) Konfiguration
# ——————————————————————————————
BATCH_SIZE = 32
EPOCHS     = 10
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "C:/Users/dasgo/OneDrive/Desktop/GitRepo/Data Science Project_Tinnitus/processed_data/train"
VAL_DIR   = "C:/Users/dasgo/OneDrive/Desktop/GitRepo/Data Science Project_Tinnitus/processed_data/val"
MODEL_DIR = "C:/Users/dasgo/OneDrive/Desktop/GitRepo/Data Science Project_Tinnitus/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ——————————————————————————————
# 2) DataLoader-Funktion
# ——————————————————————————————

def get_dataloaders(batch_size, img_size):
    # EINE FÜR TRAINING (mit Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% der Bilder spiegeln
        transforms.RandomRotation(10),           # Um bis zu 10 Grad drehen
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Helligkeit/Kontrast leicht ändern
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalisierung ist wichtig
    ])
    # EINE FÜR VALIDATION (OHNE Augmentation)
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
# 3) Modellaufbau
# ——————————————————————————————
def build_model(device):
    model = models.resnet18(weights="DEFAULT")
    in_features = model.fc.in_features
    
    # Ersetzen Sie die letzte Schicht durch eine Sequenz
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256), # Eine zusätzliche Schicht
        nn.ReLU(),                   # Aktivierungsfunktion
        nn.Dropout(0.5),             # DROPOUT: 50% der Neuronen werden zufällig deaktiviert
        nn.Linear(256, 2)            # Die finale Ausgabeschicht
    )
    return model.to(device)

# ——————————————————————————————
# 4) Trainingsroutine
# ——————————————————————————————
def train(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_acc = 0.0
    best_model_path = "" # Pfad zum besten Modell

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
        model.eval() # Modell in den Evaluationsmodus schalten (wichtig für Dropout etc.)
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad(): # Keine Gradienten berechnen, wir lernen hier nicht
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

        # — Early Stopping & Speichern des besten Modells —
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Eindeutigen Namen für das beste Modell erstellen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(MODEL_DIR, f"best_model_acc_{best_val_acc:.2%}_{timestamp}.pt")
            
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Neues bestes Modell gefunden! Acc: {best_val_acc:.2%}. Gespeichert unter: {best_model_path}\n")
        else:
            # Leerzeile für bessere Lesbarkeit, wenn sich nichts verbessert hat
            print() 

    print("\n--- Training abgeschlossen ---")
    if best_model_path:
        print(f"Bestes Modell wurde mit einer Genauigkeit von {best_val_acc:.2%} gespeichert.")
        # Lade den Zustand des besten Modells, um es zurückzugeben
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Kein Modell wurde gespeichert, da keine Verbesserung erzielt wurde.")
        
    return model
 # ———————— Ende des Trainings ————————