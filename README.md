# DropIT: <u>Drop</u>ping <u>I</u>ntermediate <u>T</u>ensors for Memory-Efficient DNN Training

A standard hardware bottleneck when training deep neural networks is GPU memory. The bulk of memory is occupied by caching intermediate tensors for gradient computation in the backward pass. We propose a novel method to reduce this footprint - Dropping Intermediate Tensors (DropIT), which drops min-k components of the intermediate tensors and approximates gradients from the sparsified tensors in the backward pass. We further propose N-sigma thresholding as a time-efficient approximation to min-k and significantly reduce computational overhead. Experiments showed that we can drop up to 80% of the elements of the intermediate tensors in fully-connected layers, saving 20\% of total GPU memory during training with no accuracy losses for Vision Transformers.

![figure1](https://user-images.githubusercontent.com/105878704/173801865-220f3938-9a52-456b-a04d-52e67fca2be7.png)


## Install

The installation of DropIT is simple. The implementation only relies on [PyTorch](https://pytorch.org/), [PyTorch-Lightning](https://www.pytorchlightning.ai/), and [timm](https://timm.fast.ai/).

```shell
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch 

pip install pytorch-lightning lightning-bolts timm

git clone https://github.com/paper3373/dropit

cd dropit
```

## Train

We provide configs in [dropit/configs](configs/). For example, training with DropIT on CIFAR-100 like:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/cifar100/vit_ti_nsigmax0.8.yaml NUM_GPUS 2
```

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/cifar100/vit_ti_parallel_minkx0.9.yaml NUM_GPUS 2
```

## Evaluation

Evaluation will be performed every N (N can be set in config, by default, N = 1) epochs during training. You can use tensorboard to see the results. An example:

```
tensorboard  --port=8888 --logdir ./outputs
```

![figure2](https://user-images.githubusercontent.com/105878704/173801881-9dc18494-6a22-4206-88c0-55ea8c6d9358.png)


To measure speed and memory, please edit and run [check.py](check.py).

## Performance

### CIFAR-100

We found PyTorch version has some minor impacts on the performance.

#### PyTorch 1.10.1:
|  Model   | Top-1 Acc | Top-5 Acc | Max Mem (GB) |
|  ----  | ----  |  ----  | ---- | 
| DeiT-Ti/16 (224x224) | 85.7 | 97.8 | 10.0 |
| DeiT-Ti/16 (224x224) w. DropIT ($N$-$\sigma$, $\gamma=0.8$) | **85.9** | **97.9** | **8.1** |

#### PyTorch 1.11.0:
|  Model   | Top-1 Acc | Top-5 Acc | Max Mem (GB) |
|  ----  | ----  |  ----  | ---- |
| DeiT-Ti/16 (224x224) | 85.4 | 97.8 | 8.6 |
| DeiT-Ti/16 (224x224) w. DropIT ($N$-$\sigma$, $\gamma=0.8$) | **85.7** | **98.1** | **6.8** |

### ImageNet-1k

#### PyTorch 1.11.0:

|  Model   | Top-1 Acc | Top-5 Acc | Max Mem (GB) |
|  ----  | ----  |  ----  | ---- |
| DeiT-Ti/16 (224x224) | 72.2 | 91.2 | 8.6 |
| DeiT-Ti/16 (224x224) w. DropIT ($N$-$\sigma$, $\gamma=0.6$) | 72.2 | 91.1 | **7.6** |
