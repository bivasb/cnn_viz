import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

num_features = 16

# Define simple CNN with 8 filters
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add weight initialization
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Stitch feature maps into grid
def create_feature_grid(feature_maps):
    # Normalize each feature map to [0, 255]
    normalized = [((fm - fm.min()) / (fm.max() - fm.min() + 1e-8) * 255).byte() 
                 for fm in feature_maps]
    
    # Create 128x128 RGB images from grayscale features
    grid_images = []
    for fm in normalized:
        # Convert to 3-channel grayscale
        img = torch.stack([fm]*3, dim=-1).cpu().numpy().astype(np.uint8)
        grid_images.append(img)
    
    # Arrange in 2 rows x 4 columns grid
    rows = []
    for i in range(0, num_features, 4):
        row = np.concatenate(grid_images[i:i+4], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    
    return Image.fromarray(grid)

# Main execution
if __name__ == "__main__":
    # Load image and model
    input_image = load_image('input.jpg')
    torch.manual_seed(42)
    model = SimpleCNN()
    
    # Forward pass
    with torch.no_grad():
        features = model(input_image)
    
    # Process and save output
    feature_maps = [features[0, i] for i in range(num_features)]
    grid_image = create_feature_grid(feature_maps)
    grid_image.save('stitched_features.jpg')
    print("Stitched feature maps saved as stitched_features.jpg")