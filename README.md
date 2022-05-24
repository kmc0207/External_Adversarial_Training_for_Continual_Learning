

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


## Pre-trained Models
To run my pre-trained model for CIFAR10, run :

```eval
python pre_trained.py
```

## Tricks
--MIR True : use MIR instead of ER
--EAT True : use EAT on MIR or ER
--NCM True : use NCM trick on MIR or ER
--RV True : use review trick on MIR or ER

## Results

Our model achieves the following performance on :


| Model name         | CIFAR10 | CIFAR100 | Mini ImageNet |
| ------------------ |---------------- | -------------- | -----------|
| ER   |     37.70%         |      17.00%      |  14.50%  |
| ER+EAT |  46.20% | 19.92%| 20.12%|

