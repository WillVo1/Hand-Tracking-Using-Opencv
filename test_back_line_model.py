"""
Test the trained back_line_model.pth on a new back image.
This script loads the trained model and predicts where the spine line should be.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# Define the same U-Net architecture used in training
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


def load_model(model_path='back_line_model.pth'):
    """Load the trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNet().to(device)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully from {model_path}")
    print(f"  - Trained for {checkpoint['epoch']+1} epochs")
    print(f"  - Best training loss: {checkpoint['loss']:.4f}")
    
    return model, device


def predict_spine_line(model, device, image_path, threshold=0.5):
    """
    Predict the spine line on a back image.
    
    Args:
        model: Trained U-Net model
        device: torch device (cpu or cuda)
        image_path: Path to input image
        threshold: Threshold for binary mask (0-1)
    
    Returns:
        original_img: Original image (RGB)
        pred_mask: Predicted mask (binary)
        pred_mask_overlay: Original image with predicted line overlay
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_size = (original_img.shape[1], original_img.shape[0])
    
    print(f"\n✓ Loaded image: {image_path}")
    print(f"  - Size: {original_size[0]}x{original_size[1]}")
    
    # Preprocess for model
    img_resized = cv2.resize(original_img_rgb, (256, 256))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    print(f"\n🔮 Running inference...")
    with torch.no_grad():
        pred_mask = model(img_tensor)
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # Threshold and resize back to original size
    pred_mask_binary = (pred_mask > threshold).astype(np.uint8) * 255
    pred_mask_resized = cv2.resize(pred_mask_binary, original_size)
    
    # Create overlay visualization
    pred_mask_overlay = original_img_rgb.copy()
    
    # Draw predicted line in red
    red_overlay = np.zeros_like(original_img_rgb)
    red_overlay[:, :, 0] = pred_mask_resized  # Red channel
    
    # Blend with original image
    mask_indices = pred_mask_resized > 127
    pred_mask_overlay[mask_indices] = [255, 0, 0]  # Red color
    
    # Also create a thicker line for better visualization
    kernel = np.ones((5, 5), np.uint8)
    pred_mask_thick = cv2.dilate(pred_mask_resized, kernel, iterations=1)
    
    pred_mask_overlay_thick = original_img_rgb.copy()
    mask_indices_thick = pred_mask_thick > 127
    pred_mask_overlay_thick[mask_indices_thick] = [255, 0, 0]  # Red color
    
    print(f"✓ Prediction complete!")
    
    return original_img_rgb, pred_mask_resized, pred_mask_overlay, pred_mask_overlay_thick


def visualize_results(original_img, pred_mask, pred_mask_overlay, pred_mask_overlay_thick, save_path='predicted_spine_line.jpg'):
    """Visualize and save the results."""
    
    # Save just the predicted spine line overlay with thick line for visibility
    cv2.imwrite(save_path, cv2.cvtColor(pred_mask_overlay_thick, cv2.COLOR_RGB2BGR))
    print(f"\n✓ Predicted spine line saved to: {save_path}")
    
    # Also save the mask alone
    cv2.imwrite('spine_line_mask.jpg', pred_mask)
    print(f"✓ Spine line mask saved to: spine_line_mask.jpg")


def main():
    """Main function to test the model."""
    print("=" * 60)
    print("BACK SPINE LINE DETECTION - MODEL TESTING")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = 'back_line_model.pth'
    IMAGE_PATH = 'back_test.png'  # The user's test image
    THRESHOLD = 0.5
    
    try:
        # Load model
        model, device = load_model(MODEL_PATH)
        
        # Run prediction
        original_img, pred_mask, pred_mask_overlay, pred_mask_overlay_thick = predict_spine_line(
            model, device, IMAGE_PATH, threshold=THRESHOLD
        )
        
        # Visualize results
        visualize_results(original_img, pred_mask, pred_mask_overlay, pred_mask_overlay_thick)
        
        print("\n" + "=" * 60)
        print("✅ TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
