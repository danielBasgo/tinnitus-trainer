# predict.py
# This script is used to predict the class of an image or all images in a directory using a pre-trained model.

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import json
from train import build_model

def get_args():
    """Defines and parses command-line arguments for the predictions script."""
    parser = argparse.ArgumentParser(description="Predict the class of an audiogram image using a trained PyTorch model.")

    # Required argument for the image to predict
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file OR a directory of images for prediction.")

    # Optional argument for the model file
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Path to the trained model file. If not provided, the latest model in the 'models/' directory will be used.")

    # Optional argument for the class mapping file
    parser.add_argument("--class_mapping",
                        type=str,
                        default="models/class_mapping.json",
                        help="Path to the JSON file that maps class indices to names.")

    # Optional argument for specifying the device
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use for inference ('auto', 'cpu', or 'cuda')."
    )

    return parser.parse_args()

def find_latest_model(model_dir="models"):
    """Finds the most recently created model file in a directory."""
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        return None
    # Find the file with the latest creation time
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def predict(image_path, model, transform, class_names, device):
    """
    Loads an image, transforms it, and returns the model's prediction and confidence.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted = output.max(1)
        predicted_class = class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item()
    return predicted_class, confidence

def main(args):
    """
    Main function to run the prediction pipeline.
    """
    # 1. Set up device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Load class mapping
    try:
        with open(args.class_mapping, 'r') as f:
            class_to_idx = json.load(f)
        # Invert the dictionary to map index to class name
        class_names = {v: k for k, v in class_to_idx.items()}
    except FileNotFoundError:
        print(f"Error: Class mapping file not found at '{args.class_mapping}'")
        return

    # 3. Find and load the model
    model_path = args.model
    if model_path is None or not os.path.isfile(model_path):
        if model_path is not None:  # If the user provided a path but it wasn't found
            print(f"Warning: Model at '{model_path}' not found.")
        print("Searching for the latest model in 'models/' directory...")
        model_path = find_latest_model()

        if model_path is None:
            print("Error: No .pt model files found in the 'models/' directory.")
            return

    try:
        # Build the model architecture
        model = build_model(num_classes=len(class_names), device=device)
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # 4. Define the same image transformations as in validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 5. Run prediction(s)
    image_path = args.image
    if os.path.isfile(image_path):
        # It's a single file
        image_paths = [image_path]
    elif os.path.isdir(image_path):
        # It's a directory, find all images inside
        print(f"Found a directory. Predicting on all images in: {image_path}")
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"Error: The provided image path is not a valid file or directory: {image_path}")
        return

    if not image_paths:
        print("No images found to predict.")
        return

    # Loop through all found images
    for single_image_path in image_paths:
        predicted_class, confidence = predict(single_image_path, model, transform, class_names, device)
        # 6. Print the result for each image
        if predicted_class is not None:
            print(f"\n--- Prediction for: {os.path.basename(single_image_path)} ---")
            print(f"  -> Predicted Class: {predicted_class}")
            print(f"  -> Confidence:      {confidence:.2%}")

if __name__ == '__main__':
    args = get_args()
    main(args)

