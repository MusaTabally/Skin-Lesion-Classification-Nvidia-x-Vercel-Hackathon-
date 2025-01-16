import csv
import shutil
import os
import random

# --- Paths ---
all_images_dir = "/home/musa-tabally/Desktop/musa_workspace/qahwaProject/data/all_images"
training_images_dir = "/home/musa-tabally/Desktop/musa_workspace/qahwaProject/data/training_images"
testing_images_dir = "/home/musa-tabally/Desktop/musa_workspace/qahwaProject/data/testing_images"
ground_truth_path = "/home/musa-tabally/Desktop/musa_workspace/qahwaProject/data/ground_truth.csv"

# --- Collect Benign and Malignant Image Lists ---
malignant_images = []
benign_images = []

with open(ground_truth_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        image_id = row[0].strip()
        label = row[1].strip()
        
        # Make sure to handle any header rows or empty lines safely:
        if not image_id or image_id.lower().startswith('isic_') is False:
            continue
        
        # Malignant
        if label == '1.0':
            malignant_images.append(image_id)
        # Benign
        elif label == '0.0':
            benign_images.append(image_id)

# --- Shuffle ---
random.shuffle(malignant_images)
random.shuffle(benign_images)

# --- Compute Train/Test Counts for Malignant ---
total_malignant = len(malignant_images)
train_malignant_count = int(total_malignant * 5/7)  # 5/7 to train
test_malignant_count = total_malignant - train_malignant_count

train_malignant = malignant_images[:train_malignant_count]
test_malignant = malignant_images[train_malignant_count:]

# --- Match Benign Counts ---
train_benign_count = train_malignant_count
test_benign_count = test_malignant_count

# Safely slice from benign list
train_benign = benign_images[:train_benign_count]
test_benign = benign_images[train_benign_count:train_benign_count + test_benign_count]

# --- Function to copy images ---
def copy_images(image_ids, source_dir, destination_dir):
    for img in image_ids:
        src = os.path.join(source_dir, img + ".jpg")
        dst = os.path.join(destination_dir, img + ".jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found.")

# --- Copy to Training Folders ---
copy_images(train_malignant, all_images_dir, training_images_dir)
copy_images(train_benign, all_images_dir, training_images_dir)

# --- Copy to Test Folders ---
copy_images(test_malignant, all_images_dir, testing_images_dir)
copy_images(test_benign, all_images_dir, testing_images_dir)

print("Done. Copied:")
print(f"  Training: {len(train_malignant)} malignant, {len(train_benign)} benign")
print(f"  Testing:  {len(test_malignant)} malignant, {len(test_benign)} benign")

