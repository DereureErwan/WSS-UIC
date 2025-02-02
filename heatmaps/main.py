import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from config import *
# from generator import *
import torch
from models import *


import matplotlib.pyplot as plt

from heatmap import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



path_slide = path_slide_tumor_test
filenames = os.listdir(path_slide)


for filename in filenames:
    img_path = os.path.join(path_slide, filename)
    image = plt.imread(img_path)
    heatmap, heatmap_treshold = evaluate(image)


    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of CNN Patch Predictions')
    plt.savefig(os.path.join(path_prediction_test, "heatmap_" + filename))

    plt.imshow(heatmap_treshold, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of CNN Patch Predictions')
    plt.savefig(os.path.join(path_prediction_test, "heatmap_treshold_" + filename))

    plt.figure(figsize=(15,8))
    plt.imshow(heatmap_treshold, cmap='hot', interpolation='nearest')
    plt.imshow(image, cmap='jet', alpha=0.8)
    plt.savefig(os.path.join(path_prediction_test, "superposition_image_heatmap_treshold_" + filename))

# plt.imshow(image)



# Example: Load an image and apply CNN on all patches

# image = plt.imread(os.path.join(path_slide, filenames[1]))

# Transform to tensor
# normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


