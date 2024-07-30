# Sign Language Digits Classification Using Simple Neural Network

## Overview
This project involves implementing a simple neural network with no hidden layers to classify sign language digits. The output layer of the neural network involves logistic regression and the sigmoid function as the activation function. The dataset contains images of sign language digits, and the goal is to predict the digit represented by the sign.

## Details
In the Jupyter notebook, I also provided mathematical reasonings and a bit of derivations for the gradient descent. The equations of cross-entropy loss and partial derivative with respect to weights and biases is then converted into python code to adjust the parameters to improve model's accuracy for prediiction.

## Dataset
The dataset used in this project is the Sign Language Digits Dataset on Kaggle. It consists of images of digits represented in sign language. The dataset is split into training and testing sets to evaluate the model's performance.

## Model Implementation
- Initialization:
  - Weights and biases are initialized
 
- Sigmoid Function:
  - The sigmoid activation function is implemented
    
- Forward and Backward Propagation:
  - Functions to compute the loss and gradients using cross-entropy loss

- Gradient Descent:
  - The model is trained using gradient descent
 
## Training and Evaluation
- Training:
  - The model is trained for a specified number of epochs with a given learning rate

- Prediction:
  - The model makes predictions on the training and testing data

- Accuracy Calculation:
  - The accuracy of the model is calculated and printed for both the training and testing datasets
 
## Results
- The model's performance is evaluated based on accuracy.
- Training and testing accuracies are printed and plotted.
 


