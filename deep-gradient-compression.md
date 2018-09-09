# Deep Gradient Compression: Reducing the communication bandwidth for distributed training ([paper](https://arxiv.org/abs/1712.01887))

## Introduction

* Gradient exchange is costly and dwarfs saving of computation time
* Network bandwidth becomes a bottleneck for scaling up distributed training, even worse on mobiles (federated learning)
* Deep Gradient Compression (DGC) solves the communication bandwidth problem by compressing the gradients

## Related Work

1. Async SGD accelerates the training by removing synchronization and updating params immediately once node completes backprop
2. Gradient Quantization to low precision values can reduce communication bandwidth (1-bit SGD, QSGD, and TernGrad). DoReFa-Net uses 1-bit weights with 2-bit gradients
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

### Gradient Sparsification

1. Gradients larger than a threshold are transmitted to reduce bandwidth
2. Rest of the gradients are accumulated locally
3. These gradients become large enough to be transmitted over time
4. Local gradient accumulation is equivalent to increasing batch size over time

### Improving local gradient accumulation

Momentum correction and local gradient clipping mitigate the problem of accuracy loss brought in by sparse updates

1. Momentum SGD:
    1. Cannot directly be applied in place of vanilla SGD (ignores discounting factor between sparse update intervals) and leads to the loss in convergence performance
    2. When the gradient sparsity is high, the update interval dramatically increases, and thus the significant side effect will harm the model performance
    2. To fix this, we locally accumulate the velocity instead of real gradient and the accumulated result is used for the subsequent sparsification and communication
2. Local gradient clipping:
    1. Widely adopted to avoid exploding gradients
    2. Rescales the gradients whenever the sum of their L2-norms exceeds a threshold
    3. Accumulates gradients over iterations on each node independently and perform the gradient clipping locally before adding the current gradient to previous accumulation
    4. Scaling the threshold by `N^−1/2`, the current node’s fraction of the global threshold if all `N` nodes had identical gradient distributions

### Overcoming the stateless effect

1. Updating of small gradients are stale and outdated
2. Staleness can slow down convergence and degrade model performance

Momentum factor masking and warm-up training mitigate staleness

1. Momentum Factor Masking:
    1. Instead of searching for a new momentum coefficient (suggested in previous works), we simply apply the same mask to both the accumulated gradients and the momentum factor
    2. Mask stops the momentum for delayed gradients, preventing the stale momentum from carrying the weights in the wrong direction

2. Warm-up Training
    1. Gradients are more diverse and aggressive in the early stages of training and sparsifying them limits the range of variation of the model
    2. The remaining accumulated gradients may outweigh the latest gradients and misguide the optimization direction
    3. Using less aggressive learning rate helps slow down the changing speed of the neural network at the start of training
    4. Less aggressive gradient sparsity helps reduce the number of extreme gradients being delayed
    5. Exponentially increasing the gradient sparsity helps the training adapt to the gradients of larger sparsity

## Experiments and Results

1. Image classification tasks
    1. The learning curve of Gradient Dropping is worse than the baseline due to gradient staleness
    2. With momentum correction, the learning curve converges slightly faster, and the accuracy is much closer to the baseline
    3. With momentum factor masking and warm-up training techniques, gradient staleness is eliminated and the learning curve closely follows the baseline
    4. Deep Gradient Compression gives 75× better compression than Terngrad with no loss of accuracy. For ResNet-50, the compression ratio is slightly lower (277× vs. 597×) with a slight increase in accuracy
2. Language modeling
    1. For language modeling, the training loss with Deep Gradient Compression closely match the baseline, so does the validation perplexity
    2. Deep Gradient Compression compresses the gradient by 462× with a slight reduction in perplexity
3. Speech recognition
    1. The learning curves show the same improvement acquired from techniques in Deep Gradient Compression as for the image network
    2. The model trained with Deep Gradient Compression gains better recognition ability on both clean and noisy speech, even when gradients size is compressed by 608×

## Performance and Conclusion

1. Hierarchically calculating the threshold significantly reduces top-k selection time for gradients
2. Deep Gradient Compression benefits even more when the communication-to-computation ratio of the model is higher and the network bandwidth is lower
3. Deep Gradient Compression (DGC) compresses the gradient by 270-600× for a wide range of CNNs and RNNs
