import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from scribble_inside_shape import Scribble

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


def get_scribbles_and_annotations_manually(filename, split):

    print("la")

    if split == "train":
        path_image = path_slide_tumor_train
    else:
        path_image = path_slide_tumor_test

    ### First extract the largest tissue component on the slide
    # to generate a healthy scribble later

    image = plt.imread(os.path.join(path_image, filename))

    # s = Scribble(filename, percent=0.0, split=split)

    # Draw tumor

    coords = []

    # fig, ax = plt.subplots()

    # ax.imshow(image)

    def on_click(event):
        # Check if the click is inside the plot area
        if event.inaxes is not None:
            # Get the x and y data from the event
            x, y = event.xdata, event.ydata
            # Print the coordinates of the clicked point
            print(f"Clicked at: x = {x}, y = {y}")
            coords.append([np.floor(x), np.floor(y)])

    # Create a plot
    fig, ax = plt.subplots()

    # Add a simple plot for reference
    ax.imshow(image)
    # Connect the click event to the on_click function
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Show the plot
    plt.legend()
    plt.show()

    # print(np.array(coords))

    if len(coords) > 0:

        scribbles_tumor = np.array(coords)
    else:
        scribbles_tumor = None

    # Draw healthy

    coords = []

    # fig, ax = plt.subplots()

    # ax.imshow(image)

    def on_click(event):
        # Check if the click is inside the plot area
        if event.inaxes is not None:
            # Get the x and y data from the event
            x, y = event.xdata, event.ydata
            # Print the coordinates of the clicked point
            print(f"Clicked at: x = {x}, y = {y}")
            coords.append([np.floor(x), np.floor(y)])

    # Create a plot
    fig, ax = plt.subplots()

    # Add a simple plot for reference
    ax.imshow(image)
    # Connect the click event to the on_click function
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Show the plot
    plt.legend()
    plt.show()

    print(np.array(coords))

    # scribbles_healthy = s.manual_scribble(np.array(coords))

    if len(coords) > 0:

        scribbles_healthy = np.array(coords)
    else:
        scribbles_healthy = None

    return (None, scribbles_tumor, None, scribbles_healthy)


class LineDrawer(object):
    lines = []

    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x, y)
        ax.figure.canvas.draw()

        self.lines.append(line)


def get_scribbles_and_annotations(filename, split):

    if split == "train":
        path_image = path_slide_tumor_train
    else:
        path_image = path_slide_tumor_test

    ### First extract the largest tissue component on the slide
    # to generate a healthy scribble later

    slide = Slide(os.path.join(path_image, filename), processed_path="")

    mask = TissueMask(
        RgbToGrayscale(),
        OtsuThreshold(),
        ApplyMaskImage(slide.thumbnail),
        GreenPenFilter(),
        RgbToGrayscale(),
        Invert(),
        OtsuThreshold(),
        RemoveSmallHoles(),
        RemoveSmallObjects(min_size=0, avoid_overmask=False),
    )
    sf = 4
    k = np.array(slide.locate_mask(mask, scale_factor=sf, outline="green"))
    k[k == 128] = 0

    contours, _ = cv2.findContours(
        k[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sizes = np.array([contours[i].shape[0] for i in range(len(contours))])
    r = np.argmax(sizes)

    annotation_healthy = contours[r] * sf

    s = Scribble(filename, percent=0.0, split=split)
    ### Extract the contours of all the tumor annotations
    dataframe_annotation = s.create_dataframe_annotations()

    ### For all the tumor annotations: generate a scribble inside the annotation
    scribbles_tumor = []
    annotations_tumor = []

    for annotation_id in tqdm(list(dataframe_annotation.columns)):
        annotation_contour = dataframe_annotation[annotation_id]
        annotation_contour = annotation_contour[~annotation_contour.isnull()]
        contour_tissue = np.vstack(annotation_contour.to_numpy())
        try:
            scribble_tumor, annotation, _, _ = s.scribble(contour_tissue)
        except:
            scribble_tumor = None
        if scribble_tumor is not (None):
            scribble_tumor = scribble_tumor
            scribble_tumor = np.expand_dims(scribble_tumor, axis=1)
            annotation = np.expand_dims(annotation, axis=1)
            annotations_tumor.append(annotation)
            scribbles_tumor.append(scribble_tumor)

    ### Generate a scribble in a healthy region
    try:
        scribble_healthy, _, _, _ = s.scribble(annotation_healthy.squeeze())
    except:
        scribble_healthy = None
    return (annotations_tumor, scribbles_tumor, annotation_healthy, scribble_healthy)
