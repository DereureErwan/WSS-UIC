import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
# from generator import *
from PIL import ImageFile
import torch
from models import *
import neptune
import argparse
from tqdm import tqdm
from torchvision import transforms


import matplotlib.pyplot as plt
import numpy as np

# Image patching function (non-overlapping patches)
def extract_patches(image, patch_size=(32, 32), stride=32):
    
    w, h = image.shape[0], image.shape[1]
    patches = []
    patch_positions = []
    
    for i in range(0, w - patch_size[0] + 1, stride):
        for j in range(0, h - patch_size[1] + 1, stride):
            top_left = [i,j]
            bottom_right = i + patch_size[0], j + patch_size[1]
            patch = image[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]
            # patch = image.crop((i, j, i + patch_size[0], j + patch_size[1]))
            patches.append(patch)
            patch_positions.append((i, j))
    
    return patches, patch_positions

# Function to create a heatmap from CNN outputs
def create_heatmap(patch_predictions, patch_positions, image_size, patch_size=(32, 32), stride=32):
    heatmap = np.zeros(image_size)  # Initialize an empty heatmap
    count_map = np.zeros(image_size)  # Keep track of how many times each pixel is included in a patch
    
    # Place the predictions back into the heatmap at their respective locations
    for i, (pred, (i, j)) in enumerate(zip(patch_predictions, patch_positions)):
        # print(pred)
        top_left = [i,j]
        bottom_right = i + patch_size[0], j + patch_size[1]
        heatmap[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]] = heatmap[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]] + pred.numpy()
        count_map[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]] += 1
    
    # Normalize the heatmap by the count map (average over all patches that cover the pixel)
    heatmap /= count_map
    return heatmap


def evaluate(image, device='cpu'):
    # image = plt.imread(img_path)

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
])

    # Create CNN model
    model = VGG16(vgg16(pretrained=False)).to(device)
    model.load_state_dict(torch.load(os.path.join(path_weights,'weights_resnet500.1')))


    ps = 128
    patches, patch_positions = extract_patches(image, patch_size=(ps, ps), stride=32)

    # Process all patches
    patch_predictions = []
    for patch in tqdm(patches):
        # Preprocess patch and pass through CNN
        patch_tensor = transform(patch).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            # print(patch_tensor.shape)
            output = model(patch_tensor)
            # print(output)
            prob = torch.softmax(output, dim=1)  # Softmax for probabilities
            # print(prob)
            predicted_class = torch.max(prob, 1)[0].item()  # Use max probability as the prediction
            predicted_class = output
        patch_predictions.append(predicted_class)

    # Generate heatmap from patch predictions
    image_size = image.shape[:-1]  # (width, height)
    heatmap = create_heatmap(patch_predictions, patch_positions, image_size, patch_size=(ps, ps), stride=32)

    # Visualize the heatmap
    

    threshold = 0.6

    heatmap_treshold = np.zeros(heatmap.shape)
    heatmap_treshold[heatmap > threshold] = heatmap[heatmap > threshold]

    return heatmap, heatmap_treshold

