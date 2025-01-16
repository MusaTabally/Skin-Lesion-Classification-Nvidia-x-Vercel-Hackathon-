# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SkinLesionDataset(Dataset):
    def __init__(self, df, image_folder, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata + 'label' column
            image_folder (str): Path to folder with image files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform
        
        # Pre-compute file paths for each row
        self.image_paths = [
            os.path.join(image_folder, f"{isic_id}.jpg")
            for isic_id in self.df['isic_id']
        ]
        
        # Define metadata columns
        self.metadata_cols = [
            'age_approx',
            'clin_size_long_diam_mm',
            'tbp_lv_areaMM2',
            'tbp_lv_area_perim_ratio',
        ]
        # Keep only columns that actually exist
        self.metadata_cols = [c for c in self.metadata_cols if c in self.df.columns]
        
        # Pre-compute metadata tensors and labels
        self.metadata_tensors = torch.tensor(
            self.df[self.metadata_cols].fillna(0.0).values,
            dtype=torch.float32
        )
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.metadata_tensors[idx], self.labels[idx]

def load_data(metadata_path, ground_truth_path, image_folder):
    """
    Load and merge metadata and ground truth data into a single DataFrame.
    Then filter so only rows whose images exist in `image_folder`.
    """
    metadata_df = pd.read_csv(metadata_path)
    groundtruth_df = pd.read_csv(ground_truth_path)
    
    # Convert groundtruth_df into a dict for fast lookup
    groundtruth_dict = dict(zip(groundtruth_df['isic_id'],
                                groundtruth_df['malignant']))
    
    def get_label(isic_id):
        # Default to 1.0 if missing, same as your existing code
        return groundtruth_dict.get(isic_id, 1.0)
    
    metadata_df['label'] = metadata_df['isic_id'].apply(get_label)
    
    # Filter to only those images actually present in the specified folder
    # List all .jpg files in that folder, then strip off the .jpg
    file_names = set(
        os.path.splitext(f)[0]
        for f in os.listdir(image_folder)
        if f.lower().endswith(".jpg")
    )
    metadata_df = metadata_df[metadata_df['isic_id'].isin(file_names)].copy()
    
    # Reset index after filtering
    metadata_df.reset_index(drop=True, inplace=True)
    
    return metadata_df

