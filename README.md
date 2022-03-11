# Deep Steal Attack published in IEEE Securityprivacy 2022

Recent advancements of Deep Neural Networks (DNNs) have seen widespread deployment in multiple security-sensitive domains. The need for resource-intensive training and use of valuable domain-specific training data have made these models a top intellectual property (IP) for model owners. One of the major threats to the DNN privacy is model extraction attacks where adversaries attempt to steal sensitive information in DNN models. In this work, we propose an advanced model extraction attack framework DeepSteal that steals DNN weights remotely for the first time with the aid of a memory side-channel attack. Our proposed DeepSteal comprises two key stages. Firstly, we develop a new weight bit information extraction method, called HammerLeak, through adopting the rowhammer based fault technique as the information leakage vector. HammerLeak leverages several novel system-level techniques tailored for DNN applications to enable fast and efficient weight stealing. Secondly, we propose a novel substitute model training algorithm with Mean Clustering weight penalty, which leverages the partial leaked bit information effectively and generates a substitute prototype of the target victim model. We evaluate the proposed model extraction framework on three popular image datasets (e.g., CIFAR-10/100/GTSRB) and four DNN architectures (e.g., ResNet-18/34/Wide-ResNet/VGG-11). The extracted substitute model has successfully achieved more than 90% test accuracy on deep residual networks for the CIFAR-10 dataset. Moreover, our extracted substitute model could also generate effective adversarial input samples to fool the victim model. Notably, it achieves similar performance (i.e., ∼1-2% test accuracy under attack) as white-box adversarial input attack (e.g., PGD/Trades). Paper Link: https://arxiv.org/pdf/2111.04625.pdf. Model Link: https://drive.google.com/drive/folders/1fphQoeKle4UYkEEwl__TO9iO0mYvT0Fb?usp=sharing


# CIFAR-10:

All the cases below are shown for 4000 rounds of hammer leak data (90 % bit information leaked).

Data from HammerLeark used for simulation:

4000 rounds/90 percent baseline: torch.tensor([0.58,0.033,0.056,0.044,0.056,0.067,0.078])
3000 rounds/80 percent baseline: torch.tensor([0.3125,0.0625,0.0625,0.0625,0.0875,0.1,0.125])
1500 rounds/60 percent baseline: torch.tensor([0.133,0.033,0.033,0.05,0.067,0.12,0.2])
Best case/over 5000 rounds: torch.tensor([0.71,0.032,0.043,0.031,0.042,0.043,0.042]) 


Command List to generate the main results of table 3:

### Resnet-18 attack performance on CIFAR-10: 

CUDA_VISIBLE_DEVICES=2 python general_v2.py --epsilon 0.031  --adv_model './results/ri.pt' --evaluate 0 --lambdas 0.00005 --layer 23 --percentage 0.9 --epochs 150

Paper Result/Expected: Accuracy = 89.05 % ; Fidelity = 91.6 %; Accuracy Under Attack = 1.94 %. 

### Resnet-18 baseline (Architecture only) performance on CIFAR-10: 

CUDA_VISIBLE_DEVICES=2 python general_v2.py --epsilon 0.031  --adv_model './results/ri.pt' --evaluate 0 --lambdas 0.0000 --layer -1 --percentage 0 --epochs 150

Paper Result/Expected: Accuracy = 73.18 % ; Fidelity = 74.29 %; Accuracy Under Attack = 61.33 %. 

### VGG11 DeepSteal: 

CUDA_VISIBLE_DEVICES=1 python vgg.py --epsilon 0.031  --adv_model './results/vgg.pt' --evaluate 0 --lambdas 0.00005 --layer 12 --percentage 0.8 --epochs 150

Paper Result/Expected: Accuracy = 84.59 % ; Fidelity = 86.24 %; Accuracy Under Attack = 16.87 %. 

### VGG11 Baseline: 

CUDA_VISIBLE_DEVICES=1 python vgg.py --epsilon 0.031  --adv_model './results/vgg.pt' --evaluate 0 --lambdas 0.0000 --layer -1 --percentage 0 --epochs 150

Paper Result/Expected: Accuracy = 70.76 % ; Fidelity = 72.06 %; Accuracy Under Attack = 61.19 %. 

### Resnet-34 DeepSteal: 

CUDA_VISIBLE_DEVICES=2 python ryad_nodel34.py --epsilon 0.031  --adv_model './results/r34.pt' --evaluate 0 --lambdas 0.00005 --layer 42 --percentage 0.9 --epochs 150

Paper Result/Expected: Accuracy = 88.17 % ; Fidelity = 89.27 %; Accuracy Under Attack = 1.44 %. 

### Resnet-34 baseline(Architecture only): 

CUDA_VISIBLE_DEVICES=2 python ryad_nodel34.py --epsilon 0.031  --adv_model './results/r34.pt' --evaluate 0 --lambdas 0.0000 --layer -1 --percentage 0.0 --epochs 150

Paper Result/Expected: Accuracy = 72.22 % ; Fidelity =   72.85 %; Accuracy Under Attack = 62.69 %. 


### Sota case (Wide ResNet):

CUDA_VISIBLE_DEVICES=1 python mwide.py --epsilon 0.031  --adv_model './results/wide.pt' --evaluate 0 --lambdas 0.00005 --layer 600 --percentage 0.9 --epochs 150
Paper Result/Expected: Accuracy = 91.93 % ; Fidelity = 93.45 %; Accuracy Under Attack = 0.05 %. 


# CIFAR-100:

### Deep Steal:

CUDA_VISIBLE_DEVICES=2 python attack_100.py --epsilon 0.031  --adv_model './results/a100.pt' --evaluate 0 --lambdas 0.00005 --layer 70 --percentage 0.9 --epochs 150

Paper Result/Expected: Accuracy = 59.8 % ; Fidelity = 64.11 %; Accuracy Under Attack = 6.61 %. 


# Only MSB cases:


### VGG: 

CUDA_VISIBLE_DEVICES=1 python mvg.py --epsilon 0.031  --adv_model './results/vgg.pt' --evaluate 0 --lambdas 0.00005 --layer 12 --percentage 0.9 --epochs 150

Paper Result/Expected: Accuracy = 81.56 % ; Fidelity = 83.33 %; Accuracy Under Attack = 18.55 %. 

### Resnet-18: 

CUDA_VISIBLE_DEVICES=2 python m18.py --epsilon 0.031  --adv_model './results/ri.pt' --evaluate 0 --lambdas 0.00005 --layer 23 --percentage 0.9 --epochs 150

Paper Result/Expected: Accuracy = 90.02 % ; Fidelity = 91.67 %; Accuracy Under Attack = 1.2 %. 


### ResNet-34:  

CUDA_VISIBLE_DEVICES=0 python m34.py --epsilon 0.031  --adv_model './results/r34.pt' --evaluate 0 --lambdas 0.00005 --layer 42 --percentage 0.9 --epochs 150

Paper Result/Expected: Accuracy = 89.92 % ; Fidelity =  91.8 %; Accuracy Under Attack = 1.03 %. 

# Error Analysis:

CUDA_VISIBLE_DEVICES=0 python error.py --epsilon 0.031  --adv_model './results/ri.pt' --evaluate 0 --lambdas 0.00005 --layer 23 --percentage 0.9 --epochs 150 --error 0.1

Paper Result/Expected: Accuracy = 86.5 % ; Fidelity =88.03 %; Accuracy Under Attack = 10.1 %. 

CUDA_VISIBLE_DEVICES=0 python error_vgg.py --epsilon 0.031  --adv_model './results/vgg.pt' --evaluate 0 --lambdas 0.00005 --layer 23 --percentage 0.9 --epochs 150 --error 0.1

Paper Result/Expected: Accuracy = 77.34 % ; Fidelity =   79.06 %; Accuracy Under Attack = 41.1 %. 