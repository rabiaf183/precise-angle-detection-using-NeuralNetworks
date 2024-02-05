#Important Libraries

import numpy as np 
import pandas as pd
import os 
from PIL import Image 
from sklearn.model_selection import train_test_split # To split the datasets into train and test
from sklearn.metrics import mean_squared_error 
from sklearn.svm import SVR # SVR (Support Vector Regression) 
import torch
import torch.nn as nn # For Neural Nets
import torch.optim as optim # For optimizations
import torch.nn.functional as F # For activation
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error 


# Function to load images and their corresponding angles from a directory and a CSV file.
def load_data(image_directory, csv_file, image_extension='.png', image_size=(64, 64)): # image_size: Resizing input image of 600x400 to 64x64 (to reduce computation time)
    df = pd.read_csv(csv_file)
    images, angles = [], []  # Initialize lists to store image data and angles

    # Iterate over each row in the df(dataframe)
    for _, row in df.iterrows():
        # Here we combined the directory path, filename, and extension to create the picture file path
        img_filename = row['filename'] + image_extension
        img_path = os.path.join(image_directory, img_filename)

        # This portion allows us to check if the image file exists at the specified path.
        if os.path.isfile(img_path):
            # Open the image, convert it to grayscale ('L'), and resize to the desired image size
            img = Image.open(img_path).convert('L').resize(image_size)
            # Convert the image to a numpy array and normalize pixel values to the range [0, 1]
            img_array = np.array(img) / 255.0
            # Append the processed image and the corresponding angles in corresponding lists
            images.append(img_array)
            angles.append(row['angle'])

    # Convert lists of images and angles to numpy arrays with type float32
    return np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)

# Specify the directory for images and csv file
image_dir = './generated_images'
csv_file = './rectangle_angles.csv'

# Call the load_data function to load images and angles
images, angles = load_data(image_dir, csv_file)

# Convert the loaded image data to PyTorch tensors, adding an additional channel dimension for compatibility with PyTorch's convolutional layers
images = torch.tensor(np.expand_dims(images, axis=1))

# Convert the angle data to a PyTorch tensor
angles = torch.tensor(angles)



# Define a CNN model class that inherits from nn.Module
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Initialize a 2D conv. layer with 1 input channel, 32 output channels, and a 3x3 kernel (since its greyscale, input chanel is 1)
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Define a max pooling layer with a 2x2 window and stride of 2 to reduce spatial dimensions.
        self.pool = nn.MaxPool2d(2, 2)
        # Initialize two additional convolutional layers with increasing number of output channels.
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)

        # Define three fully connected (FC) layers for feature reduction towards the output.
        """
        Here for the 64*64 images, first convolutional layer reduces it to 62 (64-3+1) because of 3kernel size and 1 stride. 
        The maxpooling 2*2 reduces it by half to 31. 
        Second Conv layer reduces it to first 29 and the maxpooling reduces it to 14.
        Third Conv layer reduces it to 12
        Then 64 which is the feature output is multiplied with 12*12, that gives: 12*12*64=9216
        """
        self.fc1 = nn.Linear(9216, 128) #Input to the first FC layer, output size = 128
        self.fc2 = nn.Linear(128, 64) #Input is 128 (First layer's output), output size = 64
        self.fc3 = nn.Linear(64, 1) #Input is 64 (Second layer's output), output size = 1

    # Define the forward pass of the NN
    def forward(self, x):
        # Apply the first convolutional layer followed by activation (ReLU) and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolutional layer followed by activation (ReLU) and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply the third convolutional layer followed by activation (ReLU)
        x = F.relu(self.conv3(x))
        # Flatten the output to feed into the FC layers
        x = x.reshape(x.size(0), -1)
        # Apply the first FC layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second FC layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Apply the third FC layer to produce a single output value
        x = self.fc3(x)
        # Remove any extra dimensions to get the final output
        return x.squeeze(1)

#Instantiate the feature extractor model
feature_extractor = FeatureExtractor() 

# Loss and optimizer
criterion = nn.MSELoss() #Mean Squared Error
# Define the optimizer, The Adam optimizer is chosen for its efficiency, learning rate (lr) is set to 0.001
optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)


# Split the dataset into training and test sets, test_size=0.2 means 20% for testing, 80% for training
# random seed = 42 (for reproducibility)
train_images, test_images, train_angles, test_angles = train_test_split(images, angles, test_size=0.2, random_state=42)

# Further split the training data into training and validation set, 25% of original training set, rest for training
# random seed = 42 (for reproducibility), same random seed for consistency
train_images, val_images, train_angles, val_angles = train_test_split(train_images, train_angles, test_size=0.25, random_state=42)



#Define the size of the testing, training and validation dataset
print(f'Training Data Size:{len(train_angles)}')
print(f'Test Data Size:{len(test_angles)}')
print(f'Validation Data Size:{len(val_angles)}')

# Convert the training data into a tensor dataset which can be loaded into a PyTorch DataLoader
train_dataset = torch.utils.data.TensorDataset(train_images, train_angles)

# Create a DataLoader for the training dataset. This will shuffle the data and serve it in batches of 32 samples each
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Convert the validation data into a tensor dataset
val_dataset = torch.utils.data.TensorDataset(val_images, val_angles)

# Create a DataLoader for the validation dataset. No shuffling
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize empty lists for training and validation losses
train_losses, val_losses = [], []

# Start the training loop for 200 epochs
for epoch in range(200):
    # Set the model to training mode
    feature_extractor.train()

    # Initialize a variable to accumulate the loss over an epoch
    running_loss = 0.0

    # Iterate over the training data in batches provided by train_loader
    for inputs, labels in train_loader:
        # Zero the parameter gradients. This is necessary as gradients accumulate by default.
        optimizer.zero_grad()

        # Pass the input through the model (forward pass)
        outputs = feature_extractor(inputs)

        # Compute the loss between the model output and true labels
        loss = criterion(outputs, labels)

        # Perform backpropagation to calculate the gradients
        loss.backward()

        # Update the model parameters based on the computed gradients
        optimizer.step()

        # Add the loss of the current batch to the running loss for the epoch
        running_loss += loss.item()

    # Append the average loss for this epoch to the train_losses list
    train_losses.append(running_loss / len(train_loader))


    # Start of the validation phase for each epoch
    feature_extractor.eval()
    # Set the model to evaluation mode

    # Initialize a variable to accumulate the loss over the validation dataset
    running_loss = 0.0

    # Use 'torch.no_grad()' to turn off gradients, saving memory during the forward pass
    # This is important as we do not need to update the weights during validation
    with torch.no_grad():
        # Iterate over the validation data in batches provided by val_loader
        for inputs, labels in val_loader:
            # Pass the input through the model (forward pass)
            outputs = feature_extractor(inputs)

            # Compute the loss between the model output and true labels
            loss = criterion(outputs, labels)

            # Add the loss of the current batch to the running loss for the validation phase
            running_loss += loss.item()

    # Append the average loss for the validation phase to the val_losses list
    val_losses.append(running_loss / len(val_loader))

    # Print the training and validation loss for the current epoch
    print(f'Epoch [{epoch+1}/200], Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')


# Set the feature_extractor model to evaluation mode.
feature_extractor.eval()

# Disable gradient computations, which are not needed for inference and can save memory and computation
with torch.no_grad():
    # Pass the training images through the feature_extractor model to get the learned features
    # Convert the output tensor to a NumPy array and reshape it to a 2D array where each row is a feature vector
    train_features = feature_extractor(train_images).numpy().reshape(-1, 1)

    # Similarly, pass the test images through the feature_extractor model to get their learned features
    test_features = feature_extractor(test_images).numpy().reshape(-1, 1)

# Instantiate an SVR model for regression. The parameters C=15 and epsilon=0.5 are hyperparameters of the SVR model
# C is the regularization parameter, and epsilon specifies the epsilon-tube within which no penalty is associated in the training loss function
svr = SVR(C=15, epsilon=0.5)

# Fit the SVR model to the training features and corresponding angles -- Regression
svr.fit(train_features, train_angles.numpy())

# Use the trained SVR model to predict angles for the test features
predicted_angles = svr.predict(test_features)

# Print out the actual and predicted angles for comparison
print("Actual vs Predicted Angles:")
for actual, predicted in zip(test_angles.numpy(), predicted_angles):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")




# Calculate and print the mean squared error on the test set
mse = mean_squared_error(test_angles.numpy(), predicted_angles)
print(f"Mean Squared Error on Test Set: {mse}")

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate MAE
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Calculate and print the RMSE on the test set
rmse_value = rmse(test_angles.numpy(), predicted_angles)
print(f"Root Mean Squared Error on Test Set: {rmse_value}")

# Calculate and print the MAE on the test set
mae_value = mae(test_angles.numpy(), predicted_angles)
print(f"Mean Absolute Error on Test Set: {mae_value}")

# Calculate and print the MAPE on the test set
mape_value = mape(test_angles.numpy(), predicted_angles)
print(f"Mean Absolute Percentage Error on Test Set: {mape_value:.2f}%")


# Plotting the training and validation losses
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

