import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from PIL import ImageFile
from multiprocessing import Pool
import ast
import pandas as pd
import numpy as np

# from openslide import OpenSlide
from tqdm import tqdm
import matplotlib.pyplot as plt

# ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

parser = argparse.ArgumentParser(description="Code to generate patches from scribbles.")
parser.add_argument(
    "-s",
    "--split",
    help="Select the train/test split you want to generate scribbles on .",
    type=str,
)
args = parser.parse_args()
split = args.split

if split == "train":
    path_patches = path_patches_scribbles_train
    path_images = path_slide_tumor_train
    path_dataframe = path_dataframe_train
    dataframe = pd.read_csv(path_dataframe_train, index_col=0)

else:
    path_patches = path_patches_scribbles_test
    path_images = path_slide_tumor_test
    path_dataframe = path_dataframe_test
    dataframe = pd.read_csv(path_dataframe, index_col=0)


if not os.path.exists(path_patches):
    os.makedirs(path_patches)


# #### CREATE DATASET FROM slides

slides = list(np.unique(np.array(list(dataframe["wsi"]))))


def df_to_images(filename):
    dataframe_subset = dataframe[dataframe["wsi"] == filename]
    dataframe_subset = dataframe_subset.reset_index(drop=True)
    # path = os.path.join(path_images, filename)
    img = plt.imread(os.path.join(path_images, filename))

    for idx in tqdm(range(dataframe_subset.shape[0])):
        filename, point, y = dataframe_subset.loc[idx]
        y = y.astype(int)
        point = " ".join(point.split())
        point = point.replace("[ ", "[")
        point = point.replace(" ", ",")
        point = tuple(np.array(ast.literal_eval(point)).astype(int))
        # # region = img.read_region(point, level=0, size=(ps, ps))
        # region = region.convert("RGB")

        center = point[::-1]  # Example: center at (100, 150)

        # Calculate the top-left corner of the patch
        top_left = (
            center[0] - min(center[0], ps // 2),
            center[1] - min(center[1], ps // 2),
        )

        # Calculate the bottom-right corner of the patch
        bottom_right = (
            min(center[0] + ps // 2, img.shape[0] - 1),
            min(center[1] + ps // 2, img.shape[1] - 1),
        )

        # Extract the patch from the image
        patch = img[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

        patch = np.pad(
            patch, ((ps - patch.shape[0], 0), (ps - patch.shape[1], 0), (0, 0))
        )

        plt.imsave(
            os.path.join(
                path_patches,
                "image_" + str(idx) + "_" + filename + "_" + str(y) + ".png",
            ),
            patch,
        )


if __name__ == "__main__":
    # pool = Pool(processes=16)
    # pool.map(df_to_images, slides)

    for slide in tqdm(slides):
        df_to_images(slide)
