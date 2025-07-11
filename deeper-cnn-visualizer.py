import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)

class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 128x128x16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64x64x32
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        return block1_out, block2_out

def create_feature_grid(feature_maps, cols=4):
    normalized = [((fm - fm.min()) / (fm.max() - fm.min() + 1e-8) * 255).byte() 
                 for fm in feature_maps]
    
    grid_images = []
    for fm in normalized:
        img = torch.stack([fm]*3, dim=-1).cpu().numpy().astype(np.uint8)
        grid_images.append(img)
    
    rows = []
    for i in range(0, len(grid_images), cols):
        row = np.concatenate(grid_images[i:i+cols], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    
    return Image.fromarray(grid)

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

if __name__ == "__main__":
    # Load input
    input_image = load_image('input.jpg')
    
    # Create model and get outputs
    model = DeeperCNN()
    block1_out, block2_out = model(input_image)
    
    # Process block1 features (16x128x128)
    block1_features = [block1_out[0, i] for i in range(16)]
    grid_block1 = create_feature_grid(block1_features, cols=4)
    grid_block1.save('stitched_block1.jpg')
    
    # Process block2 features (32x64x64)
    block2_features = [block2_out[0, i] for i in range(32)]
    grid_block2 = create_feature_grid(block2_features, cols=8)
    grid_block2.save('stitched_block2.jpg')
    
    print("Saved stitched_block1.jpg (16 filters) and stitched_block2.jpg (32 filters)")