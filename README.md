# ImageNet Classification with ResNet50
This project implements an ImageNet classifier using the ResNet50 architecture, achieving a target of 70% Top-1 accuracy on the ImageNet validation set. The model is trained from scratch without using pre-trained weights, ensuring a comprehensive understanding of the training process.

## About ImageNet
ImageNet is a large visual database designed for use in visual object recognition software research. It contains over 14 million images, labeled across 20,000 categories, with a subset of 1,000 categories used for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

## About ResNet50
ResNet50 is a deep residual network that consists of 50 layers. It is designed to address the vanishing gradient problem by using skip connections, allowing gradients to flow through the network more effectively. This architecture has been widely adopted for various image classification tasks due to its efficiency and accuracy. In this project, ResNet50 is utilized without pre-trained weights to train the model from scratch on the ImageNet dataset.

## Model Details
- **Architecture**: ResNet50 (without pre-trained weights)
- **Dataset**: ImageNet (1000 classes)
- **Performance**: 72.82% Top-1 accuracy in 67th epoch
- **Framework**: PyTorch


## Requirements
- Dataset downloaded into EBS gp3 using Kaggle's ImageNet1000 dataset.

## Parameters Used
- EC2 Instance: g6.12xlarge
- Training Time: 26.571 hours

## Hyperparameters
- **Batch Size**: 512 (increased from 128)
- **Learning Rate**: 0.175 (max_lr)
- **Optimizer**: SGD with momentum
- **Weight Decay**: 1e-4 (slightly reduced)
- **Number of Epochs**: 250

## Model Training Details
-**Checkpointing**: Saved the best model based on validation accuracy.
-**Mixed Precision Training**: Utilized to speed up training and reduce memory usage.
-**Learning Rate Scheduler**: Employed One Cycle Policy for learning rate scheduling. After the learning rate plateaus, used ReduceLROnPlateau scheduler.
-**Learning Rate Finder**: Used to determine the optimal learning rate before training.

## Logs
Refer logs.md

## Hugging Face Details
- The model is deployed on Hugging Face Spaces for easy access and inference.
- Live demo available at: [Hugging Face Space](https://huggingface.co/spaces/Rakavi12/ImageNet_classifier).

## Use Case
The deployed ImageNet classifier on Hugging Face Spaces allows users to easily upload images and receive predictions for the top 5 classes along with their confidence scores. This can be particularly useful for:

### How to Use
1. Visit the [Hugging Face Space](https://huggingface.co/spaces/Rakavi12/ImageNet_classifier).
2. Upload an image of an object or scene.
3. Click on the "Submit" button.
4. View the top 5 predicted classes along with their respective confidence scores.

This interface provides a user-friendly way to interact with the model and explore its capabilities.
