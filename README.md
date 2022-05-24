>📋  A template README.md for code accompanying a Machine Learning paper

# External Adversarial Training for Continual Learning

This repository is the official implementation of [My Paper Title](). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

If you want to train and evaluate on Split Mini-ImageNet, you download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/


## Training

To train the model(s) in the paper, run this command:

```train
python main.py --data CIFAR10 --per_task_epoch 1 --mem_size 1000 --EAT True
```


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python main.py --data miniimagnet --per_task_epoch 1 --EAT True
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models
To run my pre-trained model for CIFAR10, run :

```eval
python pre_trained.py
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | CIFAR10 | CIFAR100 | Mini ImageNet |
| ------------------ |---------------- | -------------- | -----------|
| ER   |     37.70%         |      17.00%      |  14.50%  |
| ER+EAT |  46.20% | 19.92%| 20.12%|

