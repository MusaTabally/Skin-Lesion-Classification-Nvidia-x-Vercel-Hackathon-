# config.py
import os

# Points to your training images folder specifically
TRAINING_IMAGES_FOLDER = "/home/musa-tabally/Desktop/musa_workspace/qahwaProject/data/training_images"

METADATA_PATH = "~/Downloads/shortestHackathonData/imageData/metadata.csv"
GROUND_TRUTH_PATH = "~/Downloads/shortestHackathonData/ISIC_2024_Training_GroundTruth.csv"

# Training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PRINT_EVERY = 100
SAVE_DIR = "checkpoints"

# Make sure save directory exists
os.makedirs(os.path.expanduser(SAVE_DIR), exist_ok=True)

