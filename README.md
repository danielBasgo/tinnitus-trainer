# Tinnitus Detection in Audiograms using Deep Learning

![Badge showing Python 3.11 in blue](https://img.shields.io/badge/Python-3.11-blue)
![Badge showing PyTorch in bright green](https://img.shields.io/badge/PyTorch-brightgreen)
![Badge showing Docker in blue](https://img.shields.io/badge/Docker-blue)

A Convolutional Neural Network (CNN) developed with PyTorch to classify audiogram images and identify patients at risk of tinnitus. The model is containerized with Docker and achieves a validation accuracy of **93.25%**.

---

## Motivation & Problem Statement

This project holds a special, personal significance for me. After recently being diagnosed with tinnitus myself, I was motivated to apply my data science skills to better understand the condition and contribute something positive to the space. My goal was to explore whether a deep learning model could be trained to automatically recognize subtle patterns in audiograms.

## Dataset

The project utilizes a combined dataset from two public sources to ensure a robust and varied training set.

*   **Data Source 1:** [Kaggle - Tinnitus Detection](https://www.kaggle.com/code/ashikshahriar/tinnitus-detection/notebook)
*   **Data Source 2:** [Kaggle - Audiological Data for Hearing Loss Classification](https://www.kaggle.com/datasets/vbookshelf/audiological-data-for-hearing-loss-classification)

**Important Note:** The raw data is **not** included in this GitHub repository to keep its size small. You must download the data manually from the sources linked above to run the project.

## Core Methodology
*   **Model Architecture:** A **ResNet18** model with Transfer Learning, customized with a **Dropout layer (p=0.5)** for regularization.
*   **Containerization:** The entire application is containerized using **Docker** and a **Conda** environment for full reproducibility.

## Results & Key Insights
After training on the combined dataset, the model achieved an outstanding performance with a **validation accuracy of 93.25%**.

A key insight came from a low-confidence prediction (58.49%) for a borderline audiogram. The model correctly identified conflicting features (normal dB levels vs. a high-frequency slope associated with tinnitus). This taught me that a robust AI model's ability to communicate its own uncertainty is a crucial feature, enabling a "human-in-the-loop" system where ambiguous cases are flagged for expert review.

## How to Run This Project

### Local Training with Conda (Recommended Start)
This method creates an exact replica of the development environment on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/danielBasgo/tinnitus-trainer.git
    cd tinnitus-trainer
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file contains all necessary dependencies. This command creates and activates an environment named `tinnitus-projekt`.
    ```bash
    conda env create -f environment.yml
    conda activate tinnitus-projekt
    ```

3.  **Download and place the data:**
    *   Download the data from **Data Source 1**. Create a folder named `audiogram_dataset` in the project root and place the `Left Ear Charts` and `Right Ear Charts` folders inside it.
    *   Download and unzip the data from **Data Source 2**. Create a folder named `new_dataset` in the project root and place the `Left ear` and `Right ear` folders from the unzipped data inside it.

    Your final folder structure should look like this:
    ```
    tinnitus-trainer/
    ├── audiogram_dataset/
    │   ├── Left Ear Charts/
    │   └── Right Ear Charts/  (from Data Source 1)
    ├── new_dataset/
    │   ├── Left ear/
    │   └── Right ear/         (from Data Source 2)
    └── ... (other project files like train.py)
    ```

4.  **Prepare the data:**
    Run the scripts sequentially to process both datasets into the `processed_data` folder.
    ```bash
    python prepare_data.py
    python prepare_new_dataset.py
    ```

5.  **Train the model:**
    ```bash
    python train.py
    ```

6.  **Make a prediction:**
    ```bash
    python predict.py --image "path/to/your/image.jpg"
    ```

## Future Work
*   **ML Deployment:** Deploy the trained model as a live API endpoint.
*   **Web App:** Create a simple web interface (e.g., using Streamlit or Flask) that allows users to upload an audiogram and receive a prediction from the model's API.

---
*This project was developed as part of my personal learning journey in data science and is motivated by my own experiences with the subject.*