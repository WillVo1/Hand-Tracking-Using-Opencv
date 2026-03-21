"""
Train with improved mask extraction - extract only the centermost spine pixels
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob

# Same U-Net architecture
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
        
        # Decoder
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


class CenterlineDataset(Dataset):
    """Extract only the centermost spine pixels from solution images."""
    
    def __init__(self, raw_dir='Raw_image', solution_dir='Solution', img_size=(256, 256)):
        self.raw_dir = raw_dir
        self.solution_dir = solution_dir
        self.img_size = img_size
        
        self.image_files = [os.path.basename(f) for f in glob.glob(os.path.join(raw_dir, "*.jpg"))]
        self.image_files.sort()
        
        valid_files = []
        for img_file in self.image_files:
            solution_path = os.path.join(solution_dir, img_file)
            if os.path.exists(solution_path):
                valid_files.append(img_file)
        
        self.image_files = valid_files
        print(f"Found {len(self.image_files)} valid paired images")
        
    def extract_centerline(self, mask):
        """Extract only the centermost pixels from the mask - the backbone line."""
        height, width = mask.shape
        
        # Find center column
        center_x = width // 2
        
        # Create a narrow vertical strip around the center
        centerline_mask = np.zeros_like(mask)
        strip_width = 30  # pixels on each side of center
        
        # For each row, find pixels within the center strip
        for y in range(height):
            row = mask[y, :]
            line_pixels = np.where(row > 127)[0]
            
            if len(line_pixels) > 0:
                # Get pixels close to center
                center_pixels = line_pixels[np.abs(line_pixels - center_x) < strip_width]
                
                if len(center_pixels) > 0:
                    # Take the median of center pixels (most centered)
                    median_x = int(np.median(center_pixels))
                    # Mark a thin line (3 pixels wide)
                    centerline_mask[y, max(0, median_x-1):min(width, median_x+2)] = 255
        
        # Apply morphological thinning to get single-pixel-wide line
        kernel = np.ones((3, 3), np.uint8)
        centerline_mask = cv2.morphologyEx(centerline_mask, cv2.MORPH_CLOSE, kernel)
        
        # Thin the line using skeletonization
        centerline_mask = cv2.ximgproc.thinning(centerline_mask)
        
        return centerline_mask
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # Load raw image
        raw_path = os.path.join(self.raw_dir, img_file)
        raw_img = cv2.imread(raw_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Load solution image
        solution_path = os.path.join(self.solution_dir, img_file)
        solution_img = cv2.imread(solution_path)
        
        # Extract line mask with low threshold
        gray = cv2.cvtColor(solution_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        
        # Extract only centerline pixels
        centerline_mask = self.extract_centerline(mask)
        
        # Resize
        raw_img = cv2.resize(raw_img, self.img_size)
        centerline_mask = cv2.resize(centerline_mask, self.img_size)
        
        # Normalize
        raw_img = raw_img.astype(np.float32) / 255.0
        centerline_mask = (centerline_mask > 127).astype(np.float32)
        
        # Convert to tensors
        raw_img = torch.from_numpy(raw_img).permute(2, 0, 1)
        centerline_mask = torch.from_numpy(centerline_mask).unsqueeze(0)
        
        return raw_img, centerline_mask


def train_model(raw_dir='Raw_image', solution_dir='Solution', epochs=20, batch_size=2, lr=0.001):
    """Train with centerline extraction."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = CenterlineDataset(raw_dir, solution_dir)
    
    if len(dataset) == 0:
        print("ERROR: No valid paired images found!")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Training images: {len(dataset)}")
    print(f"  - Mask extraction: CENTERLINE ONLY (narrow backbone)")
    
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
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {avg_loss:.6f}", end='')
        
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


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CENTERLINE-FOCUSED SPINE DETECTION TRAINING")
    print("="*60 + "\n")
    
    model = train_model(
        raw_dir='Raw_image',
        solution_dir='Solution',
        epochs=20,
        batch_size=2,
        lr=0.001
    )
    
    if model is not None:
        print(f"\n{'='*60}")
        print("✅ ALL DONE!")
        print(f"{'='*60}")
        print("\nTest with: python test_back_line_model.py")
