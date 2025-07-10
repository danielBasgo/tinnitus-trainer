# prepare_all_data.py
import os
import glob
import shutil
import sys
import stat
from sklearn.model_selection import train_test_split 

# --- 1. Configuration ---
# Source folders for the first dataset (labels from filename)
DATASET_1_SOURCES = ["audiogram_dataset/Left Ear Charts", "audiogram_dataset/Right Ear Charts"]

# Source folders for the second dataset (labels from parent directory)
DATASET_2_SOURCES = ["new_dataset/Left ear", "new_dataset/Right ear"]

# Mapping for the second dataset's folder names to our target classes
DATASET_2_MAPPING = {
    'normal': 'normal',
    'mild': 'tinnitus',
    'moderate': 'tinnitus',
    'severe': 'tinnitus',
    'profound': 'tinnitus'
}

# Destination for processed data
OUTPUT_DIR = "processed_data"

# Train/validation split ratio for the combined dataset
VAL_SPLIT_RATIO = 0.2
RANDOM_STATE = 42 # For reproducible splits

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    This is necessary for Windows, where files can sometimes be left in a read-only state.
    It changes the file permissions and retries the deletion.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

# --- 2. Main Logic ---
def main():
    print("--- Starting unified data preparation script ---")

    # Optional: Clean up the output directory for a fresh start
    if os.path.isdir(OUTPUT_DIR):
        print(f"Cleaning up old processed data in '{OUTPUT_DIR}'...")
        # Use the onerror handler to deal with permission issues on Windows
        shutil.rmtree(OUTPUT_DIR, onerror=on_rm_error)
    
    # Create the necessary directory structure
    os.makedirs(os.path.join(OUTPUT_DIR, "train", "normal"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "train", "tinnitus"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", "normal"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", "tinnitus"), exist_ok=True)
    print(f"Created fresh directory structure in '{OUTPUT_DIR}'.")

    all_files = []

    # --- Step 1: Process Dataset 1 (labels from filename) ---
    print("\n--- Processing Dataset 1 ---")
    for folder in DATASET_1_SOURCES:
        if not os.path.isdir(folder):
            print(f"Warning: Source folder not found, skipping: '{folder}'")
            continue
        
        images = glob.glob(os.path.join(folder, "*.[jJpP][pPnN][gG]*"))
        print(f"Found {len(images)} images in '{folder}'")
        for img_path in images:
            filename = os.path.basename(img_path)
            if filename.upper().startswith("N"):
                label = "normal"
            elif filename.upper().startswith("T"):
                label = "tinnitus"
            else:
                continue # Skip files that don't match N/T pattern
            all_files.append((img_path, label))

    # --- Step 2: Process Dataset 2 (labels from parent folder) ---
    print("\n--- Processing Dataset 2 ---")
    for folder in DATASET_2_SOURCES:
        if not os.path.isdir(folder):
            print(f"Warning: Source folder not found, skipping: '{folder}'")
            continue
        
        count_in_folder = 0
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_class = os.path.basename(root).lower()
                    if source_class in DATASET_2_MAPPING:
                        label = DATASET_2_MAPPING[source_class]
                        img_path = os.path.join(root, file)
                        all_files.append((img_path, label))
                        count_in_folder += 1
        print(f"Found {count_in_folder} relevant images in '{folder}'.")

    if not all_files:
        print("!!! FATAL ERROR: No images found from any data source. Stopping.", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Total images from all sources: {len(all_files)} ---")

    # --- Step 3: Split the combined dataset ---
    print(f"Splitting all images into Train ({1-VAL_SPLIT_RATIO:.0%}) and Val ({VAL_SPLIT_RATIO:.0%}) sets...")
    labels_for_split = [label for _, label in all_files]
    train_files, val_files = train_test_split(all_files, test_size=VAL_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=labels_for_split)
    print(f"Files for Training: {len(train_files)} | Files for Validation: {len(val_files)}")

    # --- Step 4: Copy files to their final destinations ---
    print("\n--- Copying files into the processed directory ---")
    def copy_files_to_dest(file_list, set_name):
        for i, (source_path, target_class) in enumerate(file_list):
            dest_folder = os.path.join(OUTPUT_DIR, set_name, target_class)
            file_extension = os.path.splitext(source_path)[1]
            # Create a unique name to avoid overwriting files
            new_filename = f"{set_name}_{target_class}_{i}{file_extension}"
            dest_path = os.path.join(dest_folder, new_filename)
            shutil.copy2(source_path, dest_path)
        print(f"Copied {len(file_list)} files to the '{set_name}' set.")

    copy_files_to_dest(train_files, 'train')
    copy_files_to_dest(val_files, 'val')

    print("\n--- ALL DONE! Your combined data is ready in 'processed_data'. ---")

if __name__ == "__main__":
    main()