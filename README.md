# Towards Constituting Mathematical Structures for Learning to Optimize

This repo is an implementation of the ICML 2023 paper "Towards Constituting Mathematical Structures for Learning to Optimize." The paper can be found here: https://arxiv.org/abs/2305.18577

## Introduction

Our work aims at solving minimization problems like 

$$\min_{x}F(x) = f(x) + r(x)$$

where $f$ is a smooth convex function and $r$ is convex and possibly nonsmooth.

**(Optimizee).** Each instance of $(f,r)$ is named as an optimizee. In our codes, we provide two types of optimizees: LASSO and Logistic+L1. We implement the gradient of $f$, the proximal operator of $r$ and the subgradient of $f+r$, for each type of optimizee. You may find them in the folder "./optimizees/".

**(Optimizer).** An iterative scheme to solve optimizees is named as an optimizer. In our codes, we provide various types of optimizers:
* Hand-designed optimizers: Adam, ISTA, FISTA, Shampoo, and subgradient descent. 
* Adaptive hyperparameter tuning: Adam-HD. 
* Algorithm unrolling: Ada-LISTA. 
* LSTM-based optimizers that are learned from data: Coordinate Blackbox LSTM, RNNprop, **Coordinate Math LSTM (our proposed)**. 

You may find them in the folder "./optimizers/".

**(Training).** Roughly speaking, we train those LSTM-based optimizers by minimizing the following loss function

$$L(\theta) = \mathbb{E}_{f,r} \sum_{k=1}^{K} f(x_k) + r(x_k)$$

In this equation, $x_k$ is the $k$-th iteration and $\theta$ represents the parameters in LSTM. Note that $x_k$ depends on $\theta$.

The horizon length, denoted as $K$, needs to be determined by hand. In the training environment, "optimizer-training-steps" in "main.py" corresponds to $K$, while in the testing environment, $K$ corresponds to "test-length".

When $K$ is large, we break down the $K$ iterations into several segments, with each segment being "unroll-length" in length. A typical setting might look something like: "optimizer-training-steps" = 100; "test-length" = 1000; "unroll-length" = 20.

Details of training cound be found in "main.py".

## Software dependencies

Our codes depend on the following packages:
```
cudatoolkit=11.3
pytorch=1.12
configargparse=1.5.3
scipy=1.10.1
```
You may install these packages in their most recent versions. However, be aware that it's essential to ensure compatibility between your CUDA and PyTorch versions.

## Start up: A toy example

One may start with LASSO of small size. Specifically, we randomly generate instances like:

$$ F(x) = \frac{1}{2}\|Ax-b\|^2_2 + \lambda \|x\|_1$$

where $A\in\mathbb{R}^{20\times40},x\in\mathbb{R}^{40},b\in\mathbb{R}^{20}$. An optimizer is then trained to find solutions for these generated optimizees.

For the training phase, you can execute the following command:
```
python main.py --config ./configs/0_lasso_toy.yaml
```

After training, the model will be stored at "./results/LASSO-toy/CoordMathLSTM.pth".

For the testing phase, run the following command:
```
python main.py --config ./configs/0_lasso_toy.yaml --test
```

The testing results can be found in "./results/LASSO-toy/losses-toy". 
Within each line, there is a float number. The value on the $k$-th line indicates the average value of the objective function at the $k$-th iteration: the value of $F(x_k)$ average across all test instances.

If your computer/server is not equipped with any GPUs, you may try the following commands:
```
python main.py --config ./configs/0_lasso_toy.yaml --device cpu
python main.py --config ./configs/0_lasso_toy.yaml --test --device cpu
```

## Reproduction

### Configurations 

Our code provides various options for optimizers and optimizees, including their respective parameters and configurations. All the configurations are listed at the beginnig of "main.py". 

We leverage the "configargparse" package to manage all these configurations. You can specify values for particular parameters either in a ".yaml" file or directly in the Python command. For instance, in the previously mentioned toy example, we set the "device" parameter to "cuda:0" in the "./configs/0_lasso_toy.yaml" file. If you add "--device cpu" to the command, the command's values will **overwrite** those in the yaml file.

To reproduce the results in our paper, you may check yaml files in "./configs/" and run commands in "./scripts/".

### Reproducing LASSO results in the paper

To obtain Figure 1, please read and run "./scripts/1_lasso_ablation.sh"

To obtain Figure 3, please read and run "./scripts/2_lasso_in_distribution.sh"

To obtain Figure 4, please read and run "./scripts/3_lasso_ood.sh"

### Reproducing Logistic results in the paper

To obtain Figure 5, please read and run "./scripts/logistic_in_distribution.sh"

To obtain Figure 6, please read and run "./scripts/logistic_ood.sh"

## Some Tips

**(Memory).** All our experiments are conducted on NVIDIA RTX A6000. If you meet memory issue (out of memory), you may reduce the batch sizes. For example, "--train-batch-size 16 --val-batch-size 16 --test-batch-size 16" 

**(Device).** If your server has multiple GPUs, you may use arguments like "--device cuda:3" to replace optimizers and optimizees to a specific device.

**(Background running).** If you want to run a command in background, you may use the command "nohup xxx >/dev/null 2>&1 &" where "xxx" means the command you want to run. All the outputs or logs can be found in the folder "./results/"

## Citing our work
If you find our codes helpful in your resarch or work, please cite our paper.

```
@inproceedings{liu2023towards,
  title={Towards Constituting Mathematical Structures for Learning to Optimize},
  author={Liu, Jialin and Chen, Xiaohan and Wang, Zhangyang and Yin, Wotao and Cai, HanQin},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
