# Large Scale Distributed Deep Networks ([paper](https://ai.google/research/pubs/pub40565))

## Introduction

* Increasing the scale of deep learning (ie. number of examples, model parameters) can increase classification accuracy drastically
* Training is slow even on GPU (because of CPU-GPU data transfer) when model size is large
* Solution is to use large scale clusters of machines to distribute training and inference in deep network

### DistBelief
* Enables model parallelism within machine (multithreading) and across machines (message passing)
* Data parallelism is achieved using model replicas
* Two novel methods implemented:

    1. Downpour SGD: Async stochastic gradient descent procedure with adaptive learning rates and supporting model replicas

    2. Sandblaster L-BFGS: Distributed implementation of [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)

## Previous Works

* For distributed gradient computation, some have relaxed the synchronization requirements exploring delayed gradient updates for convex problems
* Lock-less asynchronous stochastic gradient descent have been explored on problems with sparse gradients. Can we have best of both worlds?
* In deep learning context, most work focus on:
    * Training small model on single machine
    * Training multiple small models and ensembling them
    * Make standard deep networks parallelizable
* Current large scale computational tools:
    * MapReduce: Not suited for iterative computation in deep network training
    * GraphLabs: Would not exploit computing efficiencies available in the structured graphs

## Model Parallelism

* User defines the computation at each node in each layer of model and messages (read gradients) that need to be passed in upward (feedforward) and downward (backprop) computation phase
* Framework parallelizes computation in each machine and manages communication, synchronization and data transfer between machines during training and inference
* Performance benefits depends on connectivity structure and computational needs

## Distributed Optimization Algorithms

### Downpour SGD

1. Traditional SGD is inherently sequential which makes it impractical to apply to very large datasets
2. Its an asynchronous stochastic gradient descent using multiple replicas of Single DistBelief model
3. Divide dataset into subsets and run model on each subset
4. Model communicate updates through centralized server which keep current state of all parameters for the model
5. Asynchronous as both model and parameter shards run independently
6. **Workflow**:
    1. Model replica requests parameter server for updated copy of its model parameters
    2. Processes a mini batch to calculate gradients and sends back to server
    3. Server updates the parameters
7. Communication overhead can be reduced by limiting rate at which parameters are requested and updated
8. More robust compared to Synchronous SGD since if one machine fails, other models replicas continue processing
9. Relaxing consistency requirements found to be effective
10. Adaptive learning rate procedure greatly increases robustness
11. Use of Adagrad:
    1. Extends maximum number of model replicas that can work simultaneously
    2. Combines with `warmstarting` with single model replica before unleashing others

    This eliminates stability concerns in training.

### Sandblaster L-BFGS

1. Key idea is distributed parameter storage and manipulation
2. Provides implementation of L-BFGS
3. **Workflow**:
    1. Coordinator issues commands that can be performed by each parameter server shard independently 
    2. Results and history cache stored locally in parameter server shard
    3. This allows running large models without overhead of sending parameters and gradients to central server
4. Load balancing scheme (similar to "backup tasks" in MapReduce):
    1. Coordinator assigns each model replica small portion of work and assign replicas new work when they are free
    2. Coordinator schedules multiple copies of the outstanding portions and uses result from fastest replica
5. Contrast with Downpour SGD, Sandblaster only fetch parameters at the beginning of each batch and only send the gradients every few completed portions

## Experiments

## Conclusion
