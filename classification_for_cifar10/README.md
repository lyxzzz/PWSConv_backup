## Image classification in CIFAR10
This folder is used for image classification experiments in CIFAR10.
[register.py](python/register.py) is used for registering the dataset path.

## Requirements
- Linux or macOS with Python ≥ 3.5
- Tensorflow == 1.8.0 or 1.13.1 with a suitable CUDA version
- OpenCV

## Train the model
If the requirements are all installed and the program can be executed without any error. To run the model by using ``sh run.sh``, the core bash is implemented in [.run](.run). One can use:
```bash
pwsepsilon=0.001 ## gamma for PWS
train_model=cifar_vgg ## train model
test_model=cifar_vgg ## test model
expname=PWS ## zip name
conv_type=PWS ## conv_type, can be set as [Normal, PWS, WN]
norm_type=None ## norm_type, can be set as [None, BN, GN, LN, IN]
batch_size=256
learning_rate=10.0 ## This parameter scales all learning rates by by the corresponding value
runmodel
```
to run a model.

PWS conv is implemented in [tf_layer.py](python/utils/tf_layer.py) Line 463. For latest experimental results please refer to https://github.com/lyxzzz/PWSConv
