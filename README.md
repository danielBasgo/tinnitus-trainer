# Tinnitus Detection in Audiograms using Deep Learning

A Convolutional Neural Network (CNN) developed with PyTorch to classify audiogram images and identify patients at risk of tinnitus. The model achieves a **validation accuracy of 90.7%**.

---

## Motivation & Problem Statement

This project holds a special, personal significance for me. After recently being diagnosed with tinnitus myself, I was motivated to apply my data science skills to better understand the condition and contribute something positive to the space.

Tinnitus is a widespread auditory condition that affects the quality of life for millions of people. Early and accurate diagnosis is crucial but can be challenging. Audiograms, which are visual representations of a person's hearing ability, often contain subtle patterns that can indicate a risk of tinnitus.

My goal was to explore whether a deep learning model could be trained to automatically recognize these patterns. In doing so, I aimed not only to deepen my own knowledge but also to create a tool that could potentially assist medical professionals in diagnosis and help raise awareness of this condition.

## Dataset

The dataset for this project was sourced from the "Tinnitus Detection" notebook on Kaggle. Thank you to Ashik Shahriar for making this data available.
*   **Data Source:** [Kaggle - Tinnitus Detection](https://www.kaggle.com/code/ashikshahriar/tinnitus-detection/notebook)

The dataset consists of **1018 audiogram images**. The raw data is organized into `Right Ear Charts` and `Left Ear Charts` folders. Each image is labeled by a prefix in its filename:
*   `N... .jpg`: Normal hearing
*   `T... .jpg`: Tinnitus diagnosed

The `prepare_data.py` script processes this raw data, splits it into a training set (80%) and a validation set (20%), and organizes it into a directory structure suitable for PyTorch's `ImageFolder`.

## Methodology

This project follows a classic workflow for image classification using transfer learning.

#### 1. Data Preprocessing (`prepare_data.py`)
The raw data is copied into a `train/` and `val/` directory structure with `normal/` and `tinnitus/` subfolders, enabling efficient data loading.

#### 2. Model Architecture (`train.py`)
*   **Transfer Learning:** A **ResNet18** model, pre-trained on the ImageNet dataset, was used. This leverages the model's existing knowledge of edges, shapes, and textures.
*   **Customization:** The final fully-connected layer of the ResNet18 model was replaced with a new classifier optimized for our two classes (`normal` vs. `tinnitus`). This classifier includes:
    *   A linear layer
    *   A ReLU activation function
    *   A **Dropout layer (p=0.5)** for regularization to prevent overfitting.

#### 3. Training (`train.py`)
*   **Framework:** PyTorch
*   **Optimizer:** Adam (`lr=1e-4`)
*   **Loss Function:** Cross-Entropy Loss
*   **Epochs:** 10
*   **Data Augmentation:** To make the model more robust, random transformations were applied to the training data, including random horizontal flips, slight rotations, and color jitter.

## Results

After 10 epochs of training, the model achieved an outstanding performance, demonstrating that overfitting was successfully minimized:

| Metric         | Training Set | Validation Set   |
| -------------- | ------------ | ---------------- |
| **Accuracy**   | 90.91%       | **90.69%**       |
| **Loss**       | 0.2215       | 0.2936           |

A validation accuracy of over 90% with a minimal gap to the training accuracy indicates a robust model that generalizes well.

## How to Use

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place the data:**
    Download the raw data and place the `Right Ear Charts` and `Left Ear Charts` folders inside a directory named `audiogram_dataset/` in the project's root. The structure should look like this:
    ```
    .
    ├── audiogram_dataset/
    │   ├── Left Ear Charts/
    │   └── Right Ear Charts/
    ├── prepare_data.py
    └── train.py
    ```

4.  **Prepare the data:**
    Run the script to sort the data into the `processed_data` folder.
    ```bash
    python prepare_data.py
    ```

5.  **Train the model:**
    Start the training process.
    ```bash
    python train.py
    ```
    The trained models will be saved in the `models/` folder.

## Script Distinctions: `evaluate.py` vs `generate_confusion_matrix.py`

- **`evaluate.py`**
  - Evaluates the latest trained model on the validation dataset.
  - Prints a detailed classification report (precision, recall, F1-score) for each class in the console.
  - Displays the confusion matrix as a plot, but does **not** save it to disk.
  - Intended for quick, interactive evaluation and metric inspection.

- **`generate_confusion_matrix.py`**
  - Also evaluates the latest trained model on the validation dataset.
  - Focuses on generating and saving a high-quality confusion matrix plot as an image file (`confusion_matrix.png`).
  - Does **not** print the classification report.
  - Useful for generating figures for reports or presentations.

Both scripts share much of the same evaluation logic, but their outputs and intended use cases differ as described
    

## Future Work

- [x] ~~**Detailed Analysis:** Generate a confusion matrix and calculate Precision, Recall, and F1-Score to better evaluate error types.~~
*   **Inference Script:** Develop a script to load a single audiogram image and make a live prediction.
*   **Model Tuning:** Experiment with larger architectures (e.g., ResNet34/50) and hyperparameter optimization.
*   **Web App:** Create a simple web interface (e.g., using Streamlit or Flask) to make the model interactive.
---
*This project was developed as part of my personal learning journey in data science and is motivated by my own experiences with the subject.*
