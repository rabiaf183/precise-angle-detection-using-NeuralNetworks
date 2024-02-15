# Precise Angle Detection of Rectangles Using Neural Networks
## Introduction
The reposirory contains both the dataset file and the model file. The dataset file is used to generate a dataset of 1200 images along with a CSV file with the image name and their corresponding label since this is used for a supervised learning algorithm. The dataset generator can be modified to create more than 1200 images. And, the angle for the rectangle is varied between -10 to +10 degrees. To avoid the overfitting issues during the training, we have made the images randomly move in the display. So, the center co-ordinate for the rectangle is not uniformly distributed. The model is based on a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) combined with [MLP layers](https://en.wikipedia.org/wiki/Multilayer_perceptron) to optimize the traning.

### Learning Curve
>This graph represents our model's learning graph.

> <img src="https://github.com/rabiaf183/precise-angle-detection-using-NeuralNetworks/assets/58448531/e3c9be81-b874-47f5-bd7c-56ee1f269d8e " width="400"  align="left">

 
>To run the code, you need to install few basic dependencies.
>For python related libraries you can simply install with Pip command.
>To install Pip follow this [link](https://pip.pypa.io/en/stable/installation/).
## Installation
```
pip 
```

