import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

import config  # Our config with paths & hyperparams
from dataset import SkinLesionDataset, load_data
from model import ResNetWithMetadata

def train_model():
    # Enable cuDNN auto-tuner
    cudnn.benchmark = True
    
    # Expand user paths
    metadata_path = os.path.expanduser(config.METADATA_PATH)
    ground_truth_path = os.path.expanduser(config.GROUND_TRUTH_PATH)
    train_images_folder = os.path.expanduser(config.TRAINING_IMAGES_FOLDER)
    
    # 1. Load data from CSV, *filter* by files in the training folder
    df = load_data(metadata_path, ground_truth_path, train_images_folder)
    
    # 2. Define your training transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Create Dataset & DataLoader
    train_dataset = SkinLesionDataset(
        df=df,
        image_folder=train_images_folder,
        transform=train_transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False
    )
    
    # 4. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 5. Create model
    metadata_input_dim = len(train_dataset.metadata_cols)
    model = ResNetWithMetadata(metadata_input_dim).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 6. Set up optimizer, loss function, and mixed precision
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    # 7. Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, metadatas, labels) in enumerate(train_dataloader):
            images = images.to(device, non_blocking=True)
            metadatas = metadatas.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            with autocast():
                logits = model(images, metadatas)
                loss = criterion(logits, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
            if (batch_idx + 1) % config.PRINT_EVERY == 0:
                avg_loss = running_loss / ((batch_idx + 1) * config.BATCH_SIZE)
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
                      f"Step [{batch_idx+1}/{len(train_dataloader)}], "
                      f"Avg Loss: {avg_loss:.4f}")
        
        # End of epoch
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"==> Epoch [{epoch+1}/{config.NUM_EPOCHS}] complete, avg loss: {epoch_loss:.4f}")
        
        # Save checkpoint each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'metadata_input_dim': metadata_input_dim,
            'scaler': scaler.state_dict()
        }
        
        latest_model_path = os.path.join(config.SAVE_DIR, 'latest_model.pth')
        torch.save(checkpoint, latest_model_path, _use_new_zipfile_serialization=True)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
            torch.save(checkpoint, best_model_path, _use_new_zipfile_serialization=True)
            print(f"==> New best model saved with loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_model()
