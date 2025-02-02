# Weakly-Supervised-Segmentation for Dofus mineral segmentation

## 0. Install requirements

```pip install requirements.txt```

## 1. Donwload model weights

You can download the model weights here[https://huggingface.co/edereure/dofus]

And place them at the location `path_weights` specified in `config.py`

## 2. Run model on images

Evaluate model on images (at the location `path_slide_tumor_test` specified in `config.py`) and save predictions in `path_prediction_test` with the following command:

```python heatmaps/main.py```


