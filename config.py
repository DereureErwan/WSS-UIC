import os

path_preds = "/Users/erwandereure/Data/Dofus"
path_slide_tumor_train = os.path.join(path_preds, "train")
path_slide_tumor_test = os.path.join(path_preds, "test")
path_annotations_train = os.path.join(path_preds, "train/annotations")
path_annotations_test = os.path.join(path_preds, "test/annotations")
path_dataframe_train = os.path.join(path_preds, "dataframe_train.csv")
path_dataframe_test = os.path.join(path_preds, "dataframe_test.csv")
path_patches_scribbles_train = os.path.join(path_preds, "patches_scribbles_train")
path_patches_scribbles_test = os.path.join(path_preds, "patches_scribbles_test")
path_patches_test = os.path.join(path_preds, "patches_test")
path_patches_mask_test = os.path.join(path_preds, "patches_masks")
path_prediction_features = os.path.join(path_preds, "features_predictions")
path_slide_true_masks = os.path.join(path_preds, "truemasks")
path_uncertainty_maps = os.path.join(path_preds, "uncertainty_maps")
path_heatmaps = os.path.join(path_preds, "heatmaps")
path_segmaps = os.path.join(path_preds, "segmaps")
path_metric_tables = os.path.join(path_preds, "metric_tables")
path_weights = os.path.join(path_preds, "weights")
path_prediction_patches = os.path.join(path_preds, "patches_prediction")
path_prediction_patches_correction = os.path.join(
    path_preds, "patches_prediction_correction"
)
path_uncertainty_patches = os.path.join(path_preds, "patches_uncertainty")

path_prediction_test = os.path.join(path_preds, "path_prediction_test")


optimal_threshold = 0.6
percentage_scribbled_regions = 0.1
ov = 0.7  #### overlap
ps = 128  #### patch_size
bs = 16  #### batch_size
n_passes = 20  ## monte_carlo predictions



