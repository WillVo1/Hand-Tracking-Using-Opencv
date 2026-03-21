"""
Train a U-Net model with paired data:
- Raw_image/ folder: Clean back images (input)
- Solution/ folder: Same images with spine line drawn (target labels)
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob

# Simple U-Net for line segmentation
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, 1, 1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        return torch.sigmoid(out)


class PairedBackLineDataset(Dataset):
    """Dataset that loads paired images from Raw_image and Solution folders."""
    
    def __init__(self, raw_dir='Raw_image', solution_dir='Solution', img_size=(256, 256)):
        self.raw_dir = raw_dir
        self.solution_dir = solution_dir
        self.img_size = img_size
        
        # Get list of image filenames from raw directory
        self.image_files = [os.path.basename(f) for f in glob.glob(os.path.join(raw_dir, "*.jpg"))]
        self.image_files.sort()
        
        # Verify that corresponding solution images exist
        valid_files = []
        for img_file in self.image_files:
            solution_path = os.path.join(solution_dir, img_file)
            if os.path.exists(solution_path):
                valid_files.append(img_file)
            else:
                print(f"Warning: No solution image found for {img_file}")
        
        self.image_files = valid_files
        print(f"Found {len(self.image_files)} valid paired images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # Load raw image (input - clean back)
        raw_path = os.path.join(self.raw_dir, img_file)
        raw_img = cv2.imread(raw_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Load solution image (target - with spine line)
        solution_path = os.path.join(self.solution_dir, img_file)
        solution_img = cv2.imread(solution_path)
        
        # Extract the spine line from solution image (only very dark/black pixels - the drawn line)
        gray = cv2.cvtColor(solution_img, cv2.COLOR_BGR2GRAY)
        # Use a very low threshold (15) to only capture the actual drawn black line, not body shadows
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate mask slightly to make the line thicker for training
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Resize both images and mask
        raw_img = cv2.resize(raw_img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        # Normalize
        raw_img = raw_img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors (CHW format)
        raw_img = torch.from_numpy(raw_img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return raw_img, mask


def train_model(raw_dir='Raw_image', solution_dir='Solution', epochs=20, batch_size=2, lr=0.001):
    """Train the model with paired input/label data."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = PairedBackLineDataset(raw_dir, solution_dir)
    
    if len(dataset) == 0:
        print("ERROR: No valid paired images found!")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Training images: {len(dataset)}")
    print(f"  - Input: Clean back images from '{raw_dir}/'")
    print(f"  - Labels: Spine line images from '{solution_dir}/'")
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {avg_loss:.6f}", end='')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'back_line_model.pth')
            print(f" ✓ [Best model saved]")
        else:
            print()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved as: 'back_line_model.pth'")
    
    return model


def test_on_sample(model, device):
    """Test the model on a sample image."""
    print(f"\n{'='*60}")
    print("TESTING MODEL ON SAMPLE IMAGE")
    print(f"{'='*60}\n")
    
    model.eval()
    
    # Test on first raw image
    test_files = glob.glob('Raw_image/*.jpg')
    if len(test_files) == 0:
        print("No test images found!")
        return
    
    test_img_path = test_files[0]
    print(f"Testing on: {os.path.basename(test_img_path)}")
    
    # Load image
    test_img = cv2.imread(test_img_path)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    original_size = (test_img.shape[1], test_img.shape[0])
    
    # Preprocess
    test_img_resized = cv2.resize(test_img_rgb, (256, 256))
    test_tensor = torch.from_numpy(test_img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred_mask = model(test_tensor)
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # Threshold and resize
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask_resized = cv2.resize(pred_mask_binary, original_size)
    
    # Create overlay
    overlay = test_img_rgb.copy()
    kernel = np.ones((5, 5), np.uint8)
    pred_mask_thick = cv2.dilate(pred_mask_resized, kernel, iterations=1)
    mask_indices = pred_mask_thick > 127
    overlay[mask_indices] = [255, 0, 0]  # Red color
    
    # Save results
    cv2.imwrite('sample_test_result.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"✓ Sample test result saved as: 'sample_test_result.jpg'")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PAIRED DATA SPINE LINE DETECTION TRAINING")
    print("="*60 + "\n")
    
    # Train the model
    model = train_model(
        raw_dir='Raw_image',
        solution_dir='Solution',
        epochs=20,
        batch_size=2,
        lr=0.001
    )
    
    if model is not None:
        # Test on a sample
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_on_sample(model, device)
        
        print(f"\n{'='*60}")
        print("✅ ALL DONE!")
        print(f"{'='*60}")
        print("\nYou can now test the model on new images using:")
        print("  python test_back_line_model.py")
