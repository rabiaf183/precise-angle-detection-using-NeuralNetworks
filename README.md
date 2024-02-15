# Precise Angle Detection of Rectangles Using Neural Networks
## Introduction
The reposirory contains both the dataset file and the model file. The dataset file is used to generate a dataset of 1200 images along with a CSV file with the image name and their corresponding label since this is used for a supervised learning algorithm. The dataset generator can be modified to create more than 1200 images. And, the angle for the rectangle is varied between -10 to +10 degrees. To avoid the overfitting issues during the training, we have made the images randomly move in the display. So, the center co-ordinate for the rectangle is not uniformly distributed. The model is based on a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) combined with [MLP layers](https://en.wikipedia.org/wiki/Multilayer_perceptron) to optimize the traning.

### Learning Curve
>The graph represents our model's learning curve.

<img src="https://github.com/rabiaf183/precise-angle-detection-using-NeuralNetworks/assets/58448531/b0792525-9cf9-431a-a984-7692dac63e4d" width="400"/>
 
## Installation 
To run the code, you need to install few basic dependencies.
>most of the commands can be isntalled using pip.
>To install Pip follow this [link](https://pip.pypa.io/en/stable/installation/).
> To perform complex computation we have used numpy.

```
pip install numpy
```
> For data manipulation we have used pandas.
```
pip install pandas
```
> For machine learning tasks we have used torch. To install torch we first create a virtual environment.
 ```
sudo apt update
sudo apt install python3-venv -y
mkdir pytorch_env
cd pytorch_env
python3 -m venv pytorch_env
source pytorch_env/bin/activate
```
>Now with your virtual environment activated, you can go on to install PyTorch on your Ubuntu system.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
> For statistical modelling we have used sklearn.
```
pip install -U scikit-learn
```
> with all these dependencies you can run the following code. The model has been trained by batch gradient for about 200 epochs. Thus we get a training time of around 3 minuntes on a simple machine without GPU support.
## Results
Our model shows the MSE of around 8% which reflects the models behavior on unseen data accurately.





 


