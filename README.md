# ECS 170 - Introduction to Artificial Intelligence Project

This repo contains all the code used for this five stage project. The goal for this project is to use PyTorch to build AI models for various tasks. Our project tasks include the following: 
1. Stage 1: Environment Setup
2. Stage 2: Data Classification with MLP
3. Stage 3: Image Classification with CNN
4. Stage 4: Text Classification and Generation with RNN
5. Stage 5: Graph Embedding and Node Classification with GNN

The code for this project is organized in the following way: 
- local_code/: contains the main implementation code for all project stages, organized by experiment stage and student. The code follows a modular object-oriented design pattern where each component inherits from base classes
- script/: contains the runnable scripts that execute experiments for each stage and model, organized by student, following a similar structure to local_code/

## Stage 2: MLP

Our group used the MLP source code from the stage 1 directory provided to us to train the model for classification on handwritten digit classification. The dataset contains two pre-partitioned training and testing sets where the training set has 60,000 lines and the testing set has 10,000 lines and each line represents a labeled data instance. Each line there are 785 elements separated by commas where the first element denotes the label, holding values from {0, 1, ..., 9} and the remaining 784 elements are the features of the data intance. As a group, we tested various architectures and finalized the best models as the following: 

- **Full-batch MLP written by Jeslyn**: this model achieved a final testing accuracy of 97.55%
- **Mini-batch MLP written by Luna**: this model achieved a final testing accuracy of 98.62%

**Overal Final Model Performances:**
| MLP Model | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall |
| :-------: | :------: | :---------: | :----------------: | :-------------: |
| Full-batch | 97.55% | 0.9755 | 0.9755 | 0.9755 |
| Mini-batch | 98.62% | 0.9862 | 0.9862 | 0.9862 |

## Stage 3: CNN

In this stage of the project, our group used the following sources as our source code: 

1. [CNN with Pytorch](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
2. [ResNet with Pytorch](https://pytorch.org/hub/pytorch_vision_resnet/)

Our goal for this stage was to build CNN models to classify image data for object recognition. We had three image datasets provided to us: 

1. **MNIST**: hand-written digit images (grey-scale)
    - Training: 60000 instances
    - Testing: 10000 instances
    - Image size: 28 x 28
    - Labels: each image is associated with a single label indicating the digit number {0, 1, ..., 9}
2. **ORL**: face images (grey-scale)
    - Training: 360 instances
    - Testing: 40 instances
    - Image size: 112 x 92 x 3
    - Labels: each image is associated with a single label indicating the person {1, 2, ..., 40}
3. **CIFAR**: colored objects
    - Training: 50000 instances
    - Testing: 10000 instances
    - Image size: 32 x 32 x 3
    - Labels: each image is associated with a single label indicating the object in the label {0, 1, ..., 9}

For each of the datasets, we tested various architectures and selected the best models for each dataset: 

- **MNIST Deep Model Written by Arya**: this model achieved a final testing accuracy of 99.36%
- **ORL Deep Model Written by Arya**: this model achieved a final testing accuracy of 100%
- **ORL ResNet-18 Model Written by Jeslyn**: this model achieved a final testing accuracy of 100%
- **CIFAR-10 Deep Model Written by Luna**: this model achieved a final testing accuracy of 85.60%

**Overall Final Model Performances:**
| CNN Model | Testing Accuracy | Weighted F1 | Weighted Precision | Weighted Recall |
| :---------: | :----------------: | :-----------: | :------------------: | :---------------: |
| MNIST deep | 99.36% | 0.9936 | 0.9936 | 0.9936 |
| ORL deep | 100% | 1.000 | 1.000 | 1.000 |
| ORL ResNet | 100% | 1.000 | 1.000 | 1.000 |
| CIFAR-10 deep | 85.60% | 0.8557 | 0.8597 | 0.8560 |

## Stage 4: RNN

In stage 4 of this project, our group used the following sources as our source code: 
1. [RNN with Pytorch](https://docs.pytorch.org/docs/2.12/generated/torch.nn.RNN.html)
2. [GRU with Pytorch](https://docs.pytorch.org/docs/2.12/generated/torch.nn.GRU.html)
3. [LSTM with Pytorch](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html)

Our goal for this stage was to build RNN, GRU, and LSTM models to complete 2 tasks: 
1. **Text Classification**: perform binary classification to classify IMDB reviews into positive or negative classes 
2. **Text Generation**: generate whole jokes based on a given input of three start words

We were provided 2 datasets: 
1. **IMDB Movie Reviews**: 
    - Training: 25000 reviews
    - Testing: 25000 reviews
    - Label: Negative - score \(\le \) 4/10; Positive - score \(\ge \) 7/10
2. **Short Jokes**:
    - 1622 short jokes

For both tasks, our group built and trained RNN, GRU, and LSTM models. We tested various architectures to compare the results:
- **RNN, GRU, LSTM Text Classification Model Written by Luna**: this models all achieved final accuracies of 88.47%, 89.79%, and 89.76% respectively
- **RNN, GRU, LSTM Text Generation Model Written by Jeslyn**: evaluation of this model was by seeing whether the generated results are coherent or not (Note: Jeslyn wrote all models, and trained RNN and LSTM, Chenxuan trained the GRU model)

**Overal Final Text Classification Model Performances:**
| Model | Best Epoch | Accuracy | F1 | Precision | Recall |
| :---: | :--------: | :------: | :---: | :-------: | :-------: |
| RNN | 8 | 88.47% | 0.8847 | 0.8847 | 0.8847 |
| GRU | 11 | 89.79% | 0.8979 | 0.8979 | 0.8979 |
| LSTM | 8 | 88.76% | 0.8876 | 0.8876 | 0.8876 |

## Stage 5: GNN
