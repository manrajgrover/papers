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

## Distributed Optimization Algorithms

## Experiments

## Conclusion
