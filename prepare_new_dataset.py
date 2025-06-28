# prepare_new_dataset.py
import os
import shutil
import random

# --- Configuration ---
# The path to the root of the new, complex dataset
NEW_DATA_SOURCE_DIR = "audiogram_dataset/Right Ear Charts/right_ear"

# The destination where your combined, processed data lives
DEST_DIR = "processed_data"

# The mapping from the new dataset's subfolders to our target classes
# We use lowercase for the keys to make matching case-insensitive.
SOURCE_TO_TARGET_MAPPING = {
    'normal': 'normal',
    'mild': 'tinnitus',
    'moderate': 'tinnitus',
    'severe': 'tinnitus',
    'profound': 'tinnitus' # Add any other categories that mean 'hearing loss'
}

# The train/validation split ratio for the NEW data
VAL_SPLIT_RATIO = 0.20

# --- Main Logic ---
def main():
    print("--- Starting to process the new dataset ---")

    # Ensure the destination directories exist
    train_normal_path = os.path.join(DEST_DIR, 'train', 'normal')
    train_tinnitus_path = os.path.join(DEST_DIR, 'train', 'tinnitus')
    val_normal_path = os.path.join(DEST_DIR, 'val', 'normal')
    val_tinnitus_path = os.path.join(DEST_DIR, 'val', 'tinnitus')
    
    os.makedirs(train_normal_path, exist_ok=True)
    os.makedirs(train_tinnitus_path, exist_ok=True)
    os.makedirs(val_normal_path, exist_ok=True)
    os.makedirs(val_tinnitus_path, exist_ok=True)

    if not os.path.isdir(NEW_DATA_SOURCE_DIR):
        print(f"Error: New data source directory not found at '{NEW_DATA_SOURCE_DIR}'")
        return

    image_paths_to_process = []

    # 1. Walk through the new dataset directory to find all images and their source class
    for root, dirs, files in os.walk(NEW_DATA_SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # The parent directory name is the source class (e.g., 'Mild', 'Normal')
                source_class = os.path.basename(root).lower()
                
                # Check if this class is in our mapping
                if source_class in SOURCE_TO_TARGET_MAPPING:
                    target_class = SOURCE_TO_TARGET_MAPPING[source_class]
                    source_path = os.path.join(root, file)
                    image_paths_to_process.append((source_path, target_class))

    print(f"Found {len(image_paths_to_process)} relevant images in the new dataset.")
    if not image_paths_to_process:
        print("No images to process. Exiting.")
        return
        
    # 2. Shuffle the list to ensure a random split
    random.shuffle(image_paths_to_process)

    # 3. Split the list into training and validation sets
    split_index = int(len(image_paths_to_process) * (1 - VAL_SPLIT_RATIO))
    train_files = image_paths_to_process[:split_index]
    val_files = image_paths_to_process[split_index:]

    print(f"Splitting into {len(train_files)} training images and {len(val_files)} validation images.")

    # 4. Copy files to their final destinations
    def copy_files(file_list, set_name):
        count = 0
        for source_path, target_class in file_list:
            dest_folder = os.path.join(DEST_DIR, set_name, target_class)
            
            # Create a unique name to avoid overwriting files from the original dataset
            new_filename = f"new_dataset_{count}_{os.path.basename(source_path)}"
            dest_path = os.path.join(dest_folder, new_filename)
            
            shutil.copy2(source_path, dest_path)
            count += 1
        print(f"Copied {count} files to the '{set_name}' set.")

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    print("\n--- New dataset successfully integrated into 'processed_data' ---")


if __name__ == "__main__":
    main()