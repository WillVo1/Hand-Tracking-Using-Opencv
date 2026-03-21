"""
Check what masks are being extracted from the training data
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load one solution image
solution_img = cv2.imread('Solution/20 (2).jpg')
raw_img = cv2.imread('Raw_image/20 (2).jpg')

# Show the extraction process
gray = cv2.cvtColor(solution_img, cv2.COLOR_BGR2GRAY)
_, mask_thresh50 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
_, mask_thresh30 = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
_, mask_thresh20 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

# Try with a much lower threshold for very dark pixels (the drawn line)
_, mask_thresh10 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

print("Mask statistics with different thresholds:")
print(f"Threshold 50: {np.sum(mask_thresh50 > 0)} white pixels")
print(f"Threshold 30: {np.sum(mask_thresh30 > 0)} white pixels")
print(f"Threshold 20: {np.sum(mask_thresh20 > 0)} white pixels")
print(f"Threshold 10: {np.sum(mask_thresh10 > 0)} white pixels")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Raw Image (Input)')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Solution Image (Label)')
axes[0, 1].axis('off')

axes[0, 2].imshow(gray, cmap='gray')
axes[0, 2].set_title('Grayscale Solution')
axes[0, 2].axis('off')

axes[1, 0].imshow(mask_thresh50, cmap='gray')
axes[1, 0].set_title(f'Threshold 50 (Current)\n{np.sum(mask_thresh50 > 0)} pixels')
axes[1, 0].axis('off')

axes[1, 1].imshow(mask_thresh30, cmap='gray')
axes[1, 1].set_title(f'Threshold 30\n{np.sum(mask_thresh30 > 0)} pixels')
axes[1, 1].axis('off')

axes[1, 2].imshow(mask_thresh10, cmap='gray')
axes[1, 2].set_title(f'Threshold 10 (Very Dark)\n{np.sum(mask_thresh10 > 0)} pixels')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('mask_extraction_check.jpg', dpi=150)
print("\n✓ Visualization saved to: mask_extraction_check.jpg")
plt.show()
