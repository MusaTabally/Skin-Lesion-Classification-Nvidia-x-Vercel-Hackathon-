import tkinter as tk
from tkinter import filedialog
import os
import csv

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from model import ResNetWithMetadata  # Import the same model used in training

class TesterGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Model Tester")

        # Variables to store paths
        self.test_images_folder = None
        self.groundtruth_csv_path = None
        self.model_checkpoint_path = None

        # Buttons / Labels
        self.select_test_folder_btn = tk.Button(
            master, text="Select Test Images Folder", command=self.select_test_folder
        )
        self.select_test_folder_btn.pack(pady=5)

        self.select_groundtruth_btn = tk.Button(
            master, text="Select Ground Truth CSV", command=self.select_groundtruth_csv
        )
        self.select_groundtruth_btn.pack(pady=5)

        self.select_model_btn = tk.Button(
            master, text="Select Model Checkpoint", command=self.select_model_checkpoint
        )
        self.select_model_btn.pack(pady=5)

        self.run_test_btn = tk.Button(
            master, text="Run Test", command=self.run_test
        )
        self.run_test_btn.pack(pady=5)

        # A text widget to display logs/results
        self.text_output = tk.Text(master, height=20, width=80)
        self.text_output.pack(padx=5, pady=5)

        # Basic image transform (adjust as needed)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def select_test_folder(self):
        folder = filedialog.askdirectory(title="Select Test Images Folder")
        if folder:
            self.test_images_folder = folder
            self.write_output(f"Test folder selected: {folder}\n")

    def select_groundtruth_csv(self):
        file_path = filedialog.askopenfilename(title="Select Ground Truth CSV",
                                               filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.groundtruth_csv_path = file_path
            self.write_output(f"Ground truth CSV selected: {file_path}\n")

    def select_model_checkpoint(self):
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint", 
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_checkpoint_path = file_path
            self.write_output(f"Model checkpoint selected: {file_path}\n")

    def run_test(self):
        """Loads the model, iterates over test images, compares to ground truth, 
        prints results, and prints overall accuracy."""
        if not self.test_images_folder or not self.groundtruth_csv_path or not self.model_checkpoint_path:
            self.write_output("Error: Please select test folder, ground truth CSV, and model checkpoint first.\n")
            return

        # 1. Load ground truth into a dict: { 'ISIC_0012345': 0 or 1, ... }
        groundtruth_dict = {}
        with open(self.groundtruth_csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip empty lines or header
                if not row or row[0].startswith("isic_id"):
                    continue
                isic_id = row[0].strip()
                label = float(row[1])  # 0.0 or 1.0
                groundtruth_dict[isic_id] = label

        # 2. Load the model checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.write_output(f"Using device: {device}\n")

        # Build the same model structure used in training
        checkpoint = torch.load(self.model_checkpoint_path, map_location=device)
        metadata_input_dim = checkpoint["metadata_input_dim"]  # Retrieve metadata_input_dim from checkpoint
        model = ResNetWithMetadata(metadata_input_dim=metadata_input_dim)
        model.to(device)

        model.load_state_dict(checkpoint["model_state_dict"])  # Load weights
        model.eval()  # inference mode

        # 3. Iterate over each image in the test folder
        correct = 0
        total = 0

        # We’ll assume images are named e.g. ISIC_0123456.jpg
        # so we can parse the base name to find the isic_id
        for file_name in os.listdir(self.test_images_folder):
            if not file_name.lower().endswith(".jpg"):
                continue

            isic_id = os.path.splitext(file_name)[0]

            # If not in groundtruth_dict, skip or assume label=?
            if isic_id not in groundtruth_dict:
                self.write_output(f"Warning: {isic_id} not found in ground truth CSV. Skipping.\n")
                continue

            true_label = groundtruth_dict[isic_id]  # 0.0 or 1.0

            # Load image
            image_path = os.path.join(self.test_images_folder, file_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(device)  # shape [1,3,224,224]

            # Create dummy metadata (all zeros) for testing
            meta_tensor = torch.zeros((1, metadata_input_dim), dtype=torch.float32).to(device)

            # Forward pass
            with torch.no_grad():
                logit = model(image_tensor, meta_tensor)
                prob = torch.sigmoid(logit).item()
            
            # Pred threshold
            pred_label = 1.0 if prob >= 0.5 else 0.0
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            total += 1

            # Print result for this image
            self.write_output(
                f"Image: {file_name}, "
                f"Pred: {pred_label:.0f}, "
                f"True: {true_label:.0f}, "
                f"{'Correct' if is_correct else 'Wrong'}\n"
            )

        # 4. Final accuracy
        accuracy = (correct / total) * 100 if total > 0 else 0
        self.write_output(f"\nOverall accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")

    def write_output(self, text):
        """Helper to write text to the Text widget (and auto-scroll)."""
        self.text_output.insert(tk.END, text)
        self.text_output.see(tk.END)

# --------------------------
# Main entry point
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TesterGUI(root)
    root.mainloop()
