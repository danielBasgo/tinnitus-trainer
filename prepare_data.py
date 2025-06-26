# prepare_data.py (BULLETPROOF VERSION)
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# --- 1. DEFINE YOUR FOLDERS ---
SOURCE_FOLDERS = ["audiogram_dataset/Left Ear Charts", "audiogram_dataset/Right Ear Charts"] 
OUTPUT_FOLDER = "processed_data"

# --- 2. THE SCRIPT ---
print("--- Step 1: Finding all original images ---")

all_files = []
for folder in SOURCE_FOLDERS:
    if not os.path.isdir(folder):
        print(f"!!! FATAL ERROR: Cannot find folder '{folder}'. Is it in the right place?")
        exit()
    
    found_images = glob.glob(os.path.join(folder, "*.[jJpP][pPnN][gG]*"))
    all_files.extend(found_images)
    print(f"Found {len(found_images)} images in '{folder}'")

if not all_files:
    print("!!! FATAL ERROR: Found 0 total images. Nothing to do. Stopping.")
    exit()

print(f"\n--- Step 2: Splitting {len(all_files)} images into Train (80%) and Val (20%) sets ---")
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"Files for Training: {len(train_files)}")
print(f"Files for Validation: {len(val_files)}")


# --- STEP 3: CREATE FOLDERS IF THEY DON'T EXIST (NO DELETING!) ---
print("\n--- Step 3: Ensuring output folders exist ---")
# The "exist_ok=True" part prevents errors if the folder is already there.
# This is our fix for the PermissionError.
os.makedirs(os.path.join(OUTPUT_FOLDER, "train", "normal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "train", "tinnitus"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "val", "normal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "val", "tinnitus"), exist_ok=True)
print("'processed_data' folder is ready.")


print("\n--- Step 4: Copying files into the folders ---")

def copy_files_to_destination(file_list, set_name):
    count = 0
    for file_path in file_list:
        filename = os.path.basename(file_path)
        
        if filename.startswith("N"):
            label = "normal"
        elif filename.startswith("T"):
            label = "tinnitus"
        else:
            continue

        destination = os.path.join(OUTPUT_FOLDER, set_name, label, filename)
        shutil.copy2(file_path, destination)
        count += 1
    print(f"Copied/Updated {count} files to the '{set_name}' set.")

copy_files_to_destination(train_files, "train")
copy_files_to_destination(val_files, "val")

print("\n--- ALL DONE! Your data is ready. ---")