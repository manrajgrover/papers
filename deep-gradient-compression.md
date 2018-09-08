# Deep Gradient Compression: Reducing the communication bandwidth for distributed training ([paper](https://arxiv.org/abs/1712.01887))

## Introduction

* Gradient exchange is costly and dwarfs saving of computation time
* Network bandwidth becomes bottleneck for scaling up distributed training, even worse on mobiles (federated learning)
* Deep Gradient Compression (DGC) solves the communication bandwidth problem by compressing the gradients

## Related Work

1. Async SGD accelerates the training by removing synchronization and updating params immediately once node completes backprop
2. Gradient Quantization to low precision values can reduce communication bandwidth (1-bit SGD, QSGD and TernGrad). DoReFa-Net uses 1-bit weights with 2-bit gradients
3. Gradient Sparsification can be done in many ways:
    1. Threshold quantization to send gradients larger than predefined constant
    2. Choose a fixed proportion of positive and negative gradient updates
    3. Gradient dropping to sparsify gradients by a single threshold based on the absolute value (required adding layer normalization)
    4. Automatically tunes the compression rate depending on local gradient activity, and gained high compression ratio with negligible degradation of accuracy
4. Compared to previous work, Deep Gradient Compression:
    1. Pushes gradient compression ratio
    2. Does not require altering model structure
    3. Results in no loss in accuracy

## Deep Gradient Compression
