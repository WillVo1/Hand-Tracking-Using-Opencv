"""
Train a U-Net model to detect and draw a line down someone's back.
This extracts the black line annotations from training images and trains a segmentation model.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image

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


class BackLineDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
        print(f"Found {len(self.image_paths)} training images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract black line as mask (black pixels with low RGB values)
        # The line is dark/black, so we threshold to find it
        gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate mask slightly to make it more visible
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors (CHW format)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return img, mask


def train_model(data_dir='data', epochs=50, batch_size=2, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = BackLineDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining for {epochs} epochs...")
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'back_line_model.pth')
            print(f"  -> Saved best model (loss: {best_loss:.4f})")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print("Model saved as 'back_line_model.pth'")
    return model


if __name__ == "__main__":
    # Train the model
    model = train_model(data_dir='data', epochs=30, batch_size=2, lr=0.001)
    
    # Test on a sample image
    print("\nTesting model on sample image...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_img = cv2.imread('data/20 (14).jpg')
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img_resized = cv2.resize(test_img_rgb, (256, 256))
    test_tensor = torch.from_numpy(test_img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_mask = model(test_tensor)
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # Visualize result
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (test_img.shape[1], test_img.shape[0]))
    
    # Save test result
    cv2.imwrite('test_prediction.jpg', pred_mask)
    print("Test prediction saved as 'test_prediction.jpg'")
