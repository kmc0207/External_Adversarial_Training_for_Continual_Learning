# External_Adversarial_Training_for_Continual_Learning

Please download requirement.txt
pip install -r requirement.txt


To test our methods, you can use main.py

ER :
main.py --data CIFAR10 --per_task_epoch 1 

EAT :
main.py --data CIFAR100 --per_task_epoch 1 --vir_train True
