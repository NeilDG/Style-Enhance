# Mobile Image Enhancement through Image Style Transfer
 
This repository contains the code for the deep learning approach to image enhancement through image style transfer which takes an image from a low end smartphone and enhances it in the style of a higher end device.

## Sample Images
TO DO

## Setup

1. Download the pre-trained [VGG-19 model](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing) and put it into `vgg_pretrained/` folder.
2. Download the [mobile image pair dataset (MIPD)](https://drive.google.com/open?id=1CmDvlpZbZuoVQ8keSA-oIaZgVIo2ueI7) and extract it on the `images/` folder.

## Enhance Images

Run `test_model.ipynb` using Jupyter Notebook or Jupyter Lab.

Parameters and their default values:
* `phone`: `Nova2i` - Input device
* `model`: `StyleEnhance` - Name of the model
* `iteration`: `27239` - Number of iterations for the trained model
- `gt`: `iPhone8` - Reference ground truth device
- `res`: `iPhone8_resize` - Resolution
- `resnet`: `16` - Number of residual blocks (The mobile implementation has 4 residual blocks by default)

Alternatively you can run `test_model.ipynb` using runipy with the following command.
```bash
runipy test_model.ipynb
```


## Train Model
Run `train_model.ipynb` using Jupyter Notebook or Jupyter Lab.

Parameters and their default values:
- `source`: `Nova2i` - Input device
- `target`: `iPhone8` - Reference ground truth device
- `model`: `StyleEnhance` - Name of the model
- `batch_size`: `16` - Number of epochs to run
- `load_step`: `1000` - Number of batches to load in the memory. Set to `-1` to load the entire training dataset.
`epochs`: `10` - Number of epochs to run
`resnet`: `16` - Number of residual blocks (The mobile implementation has 4 residual blocks by default)

Alternatively you can run `train_model.ipynb` using runipy with the following command.
```bash
runipy train_model.ipynb
```
