# Skin Lesion Classification Using ResNet with Metadata

## Overview
This project implements a machine learning pipeline for classifying skin lesions as benign or malignant based on both image data and associated metadata. The solution utilizes a convolutional neural network (CNN) based on ResNet architecture, augmented with a metadata processing component to enhance predictive performance. It supports training, testing, and deployment, with a FastAPI-based REST API for real-time predictions and a GUI for evaluation.

---

## Features

- **Data Preprocessing:**
  - Organizes and partitions image data into training and testing sets.
  - Incorporates metadata such as age and lesion measurements to improve classification accuracy.

- **Model Architecture:**
  - Utilizes ResNet18 pre-trained on ImageNet for feature extraction from images.
  - Augments ResNet with a metadata-processing Multi-Layer Perceptron (MLP) for combined analysis.

- **Training:**
  - Supports mixed precision training for faster performance.
  - Includes hyperparameter tuning options for learning rate, batch size, and epochs.

- **Testing and Evaluation:**
  - A GUI application for selecting and testing images against ground truth.
  - Displays classification accuracy and detailed logs.

- **Deployment:**
  - A REST API powered by FastAPI for real-time prediction.
  - Provides confidence scores for predictions.

---

## Project Structure

- **`config.py`**: Contains configuration parameters for paths, hyperparameters, and model saving.

- **`dataPartition.py`**: Handles partitioning of image data into training and testing sets based on ground truth labels.

- **`dataset.py`**: Defines the `SkinLesionDataset` class, which loads and preprocesses images and metadata for model input.

- **`model.py`**: Implements the `ResNetWithMetadata` architecture for combining image and metadata features.

- **`train.py`**: Script for training the model, including loss tracking and checkpoint saving.

- **`test.py`**: GUI application for evaluating model performance on test datasets.

- **`main.py`**: FastAPI application for serving the trained model via REST endpoints.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch
- FastAPI
- torchvision
- PIL (Pillow)
- Tkinter

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepositoryName.git
   cd YourRepositoryName
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Adjust paths in `config.py` to match your local dataset and metadata file locations.

4. Prepare the dataset by running:
   ```bash
   python dataPartition.py
   ```

---

## Usage

### Training
To train the model, run the `train.py` script:
```bash
python train.py
```
Trained models will be saved in the directory specified in `config.SAVE_DIR`.

### Testing
Launch the GUI application to test the model:
```bash
python test.py
```

### Deployment
Run the FastAPI application to serve the model:
```bash
python main.py
```
The API will be available at `http://127.0.0.1:8000`.

#### API Endpoint
- **POST** `/predict`
  - Upload an image to classify it as benign or malignant.
  - Returns the result and confidence score.

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path/to/image.jpg"
```

---

## Results
- **Accuracy:** Achieved `XX%` accuracy on the test dataset.
- **Model Insights:** Effective integration of metadata significantly improves classification performance.

---

## Future Work
- Support for additional metadata fields.
- Expand testing to include external datasets.
- Deploy model on cloud platforms for scalability.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- Pretrained ResNet model from PyTorch.
- ISIC Skin Lesion Dataset for training and evaluation.

For any inquiries or contributions, please contact [Your Email Address].

