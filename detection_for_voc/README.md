## Object Detection in VOC and COCO
This folder is used for SSD experiments in VOC and COCO.
[register.py](python/register.py) is used for registering the dataset path.

1. To use VOC dataset, one should revise [voc_preprocess](python/dataloader_voc/voc_preprocess.py) with a corresponding path:
```python
img_root_dir = 'dataset/VOC/VOC{}/VOC{}/JPEGImages'.format(voc_type, voc_date)
xml_root_dir = 'dataset/VOC/VOC{}/VOC{}/Annotations'.format(voc_type, voc_date)
```
in Line 19-20.
2. run ``python python/dataloader_voc/voc_preprocess.py`` to generate the csv file used for ``register.py``.
3. change the path in [register.py](python/register.py) to a correct path.

## Requirements
- Linux or macOS with Python ≥ 3.5
- Tensorflow == 1.8.0 or 1.13.1 with a suitable CUDA version
- OpenCV

## Train the model
If the requirements are all installed and the program can be executed without any error. To run the model by using ``sh run.sh``, the core bash is implemented in [.run](.run). One can use:
```bash
pwsepsilon=0.001 ## gamma for PWS
last_norm=None 
expname=PWS ## zip name
conv_type=PWS ## conv_type, can be set as [Normal, PWS, WN]
norm_type=None ## norm_type, can be set as [None, BN, GN, LN, IN]
batch_size=32
learning_rate=10.0 ## This parameter scales all learning rates by by the corresponding value
dataset=voc 
train_model=${dataset}_ssd ## train model
test_model=${dataset}_ssd ## test model
runmodel
```
to run a model.

PWS conv is implemented in [tf_layer.py](python/utils/tf_layer.py) Line 429. 
