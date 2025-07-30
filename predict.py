# predict.py
# This script is used to predict the class of an image or all images in a directory using a pre-trained model.

import argparse
import json
import os
import glob
from typing import List, Tuple, Dict, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image

from train import build_model

def get_args() -> argparse.Namespace:
    """Defines and parses command-line arguments for the predictions script."""
    parser = argparse.ArgumentParser(description="Predict the class of an audiogram image using a trained PyTorch model.")

    # Required argument for the image to predict
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file OR a directory of images for prediction.")

    # Optional argument for the model file
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Path to the trained model file (*.pt). If not provided, the latest model in the 'models/' directory will be used.")

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

def find_latest_model(model_dir: str = "models") -> Optional[str]:
    """Finds the most recently created model file (*.pt) in a directory."""
    if not os.path.isdir(model_dir):
        return None
    list_of_files = glob.glob(os.path.join(model_dir, '*.pt'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def predict(image_path: str, model: torch.nn.Module, transform: transforms.Compose, class_names: Dict[int, str], device: torch.device) -> Optional[Tuple[str, float]]:
    """
    Loads an image, transforms it, and returns the model's prediction and confidence.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted = output.max(1)
        predicted_class = class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item()
    return predicted_class, confidence

def load_model_for_inference(args: argparse.Namespace) -> Optional[Tuple[torch.nn.Module, Dict[int, str], torch.device]]:
    """Handles device setup, model loading, and class mapping."""
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        with open(args.class_mapping, 'r') as f:
            class_to_idx = json.load(f)
        class_names = {v: k for k, v in class_to_idx.items()}
    except FileNotFoundError:
        print(f"Error: Class mapping file not found at '{args.class_mapping}'")
        return None

    model_path = args.model
    if model_path is None or not os.path.isfile(model_path):
        if model_path is not None:
            print(f"Warning: Model at '{model_path}' not found.")
        print("Searching for the latest model in 'models/' directory...")
        model_path = find_latest_model()

        if model_path is None:
            print("Error: No .pt model files found in the 'models/' directory.")
            return None

    try:
        print(f"Loading model: {os.path.basename(model_path)}")
        model = build_model(num_classes=len(class_names), device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, class_names, device
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return None
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

def main(args: argparse.Namespace):
    """Main function to run the prediction pipeline."""
    # 1. Load model, class names, and device
    load_result = load_model_for_inference(args)
    if load_result is None:
        return # Exit if setup failed
    model, class_names, device = load_result

    # 4. Define the same image transformations as in validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 5. Find image(s) to predict
    image_path = args.image
    if os.path.isfile(image_path):
        # It's a single file
        image_paths = [image_path]
    elif os.path.isdir(image_path):
        # It's a directory, find all images inside
        print(f"\nFound a directory. Predicting on all images in: {image_path}")
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"Error: The provided image path is not a valid file or directory: {image_path}")
        return

    if not image_paths:
        print("No images found to predict.")
        return

    # 6. Loop through all found images and predict
    total_images = len(image_paths)
    print(f"Found {total_images} image(s) to predict.")
    for single_image_path in image_paths:
        predicted_class, confidence = predict(single_image_path, model, transform, class_names, device)
        # 7. Print the result for each image
        if predicted_class is not None:
            print(f"\n--- Prediction for: {os.path.basename(single_image_path)} ---")
            print(f"  -> Predicted Class: {predicted_class}")
            print(f"  -> Confidence:      {confidence:.2%}")
    print("\n--- Prediction script finished. ---")

if __name__ == '__main__':
    args = get_args()
    main(args)
