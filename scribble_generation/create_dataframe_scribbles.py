import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from scribble_inside_shape import *
import warnings
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from multiprocessing import Pool
from utils import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Code to generate the dataframe of the scribbles."
)
parser.add_argument(
    "-s",
    "--split",
    help="Select the train/test split you want to generate scribbles on .",
    type=str,
)

parser.add_argument(
    "-p",
    "--percentage_scribbled_regions",
    help="Select the percentage of scribble regions to scribble",
    type=float,
    default=percentage_scribbled_regions,
)
args = parser.parse_args()
split = args.split
percentage_scribbled_regions = args.percentage_scribbled_regions

if split == "train":
    path_slide = path_slide_tumor_train
    filenames = os.listdir(path_slide)
    path_dataframe = path_dataframe_train
else:
    path_slide = path_slide_tumor_test
    filenames = os.listdir(path_slide)
    path_dataframe = path_dataframe_test

dic = {}

print("hey")
for i, filename in enumerate(tqdm(os.listdir(path_slide))):
    ### Go through all the slides to create healthy and tumor scribbles

    path_image = os.path.join(path_slide, filename)
    (
        annotations_tumor,
        scribbles_tumor,
        annotation_healthy,
        scribbles_healthy,
    ) = get_scribbles_and_annotations_manually(path_image, split)
    dic[filename] = [scribbles_tumor, scribbles_healthy, annotations_tumor]

    ###### SELECT percentage_scribbled_regions% of tumor regions ######
    ###### Remove scribble Healthy on tumor regions ######

    # if dic[filename][1] is not None:
    #     n = dic[filename][1].shape[0]
    #     bool_filename = np.ones(n)
    #     n_tumor = len(dic[filename][0])
    #     areas = np.zeros(n_tumor)

    #     for i in range(n):
    #         point = Point(dic[filename][1][i])
    #         for j, polygon in enumerate(dic[filename][2]):
    #             poly = Polygon(np.squeeze(polygon))
    #             if i == 0:
    #                 areas[j] = poly.area
    #             if poly.contains(point):
    #                 bool_filename[i] = 0
    #                 break
    #     dic[filename][1] = dic[filename][1][bool_filename.astype(bool)]

    #     args = np.flip(np.argsort(areas))[: int(percentage_scribbled_regions * n_tumor)]
    #     dic[filename][0] = [dic[filename][0][u] for u in args]


df = pd.DataFrame(columns=["wsi", "point", "class"])

for filename in tqdm(dic.keys()):
    scribble_tumor, scribble_healthy, _ = dic[filename]
    # scribble_tumor = np.squeeze(np.concatenate(scribble_tumor))
    # scribble_healthy = np.squeeze(np.concatenate(scribble_healthy))

    if scribble_tumor is not None:
        # scribble_tumor = np.squeeze(np.concatenate(scribble_tumor))

        for i in range(scribble_tumor.shape[0]):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{"wsi": filename, "point": scribble_tumor[i], "class": 1}]
                    ),
                ],
                ignore_index=True,
            )

        # df = df.append(
        #     {"wsi": filename, "point": scribble_tumor[i], "class": 1}, ignore_index=True
        # )
    if scribble_healthy is not None:
        # scribble_healthy = np.squeeze(np.concatenate(scribble_healthy))
        for i in range(scribble_healthy.shape[0]):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{"wsi": filename, "point": scribble_healthy[i], "class": 0}]
                    ),
                ],
                ignore_index=True,
            )
            # df = df.append(
            #     {"wsi": filename, "point": scribble_healthy[i], "class": 0},
            #     ignore_index=True,
            # )
if not os.path.exists(path_dataframe):
    df.to_csv(path_dataframe)
else:
    df_old = pd.read_csv(path_dataframe)
    # df_old.drop(df_old.columns[len(df.columns) - 1], axis=1, inplace=True)
    df_merged = pd.concat(
        [
            df,
            df_old,
        ],
        ignore_index=True,
    )
    df_merged = df_merged[["wsi", "point", "class"]]
    # df_merged = df.append(df_old, ignore_index=True)
    df_merged.to_csv(path_dataframe)
