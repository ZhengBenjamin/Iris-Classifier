import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Data():
  """
  Class to handle data processing
  """
  
  @staticmethod
  def gen_data(data: str) -> None:
    """
    Generates data vectors for all iris data points
    """
    data = pd.read_csv(data)
    data['species'] = data['species'].map({"setosa": 0, "versicolor": 1, "virginica": 2}) # Replace species with numerical values
    
    x = data.drop('species', axis=1).values # Input 
    y = data['species'].values # Output
    
    return x, y
    
class Model(nn.Module):
  """
  Class to handle model creation and training
  """
  
  def __init__(self, x, y, device):
    super(Model, self).__init__()
    
    scaler = StandardScaler() 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 30-70 split for test / train data

    x_train = scaler.fit_transform(x_train) # Fit and transform training data
    x_test = scaler.transform(x_test) # Transform test data
    
    # Convert to torch tensors and move to device
    self.x_train = torch.FloatTensor(x_train).to(device)
    self.x_test = torch.FloatTensor(x_test).to(device)
    self.y_train = torch.LongTensor(y_train).to(device)
    self.y_test = torch.LongTensor(y_test).to(device)
    
    self.input = nn.Linear(len(x[0]), 64)
    self.layer1 = nn.Linear(64, 32)
    self.output = nn.Linear(32, 3)
    
    # self.input = nn.Linear(len(x[0]), 256)
    # self.layer1 = nn.Linear(256, 128)
    # self.layer2 = nn.Linear(128, 64)
    # self.layer3 = nn.Linear(64, 32)
    # self.layer4 = nn.Linear(32, 16)
    # self.output = nn.Linear(16, 3)
    
    # self.relu = nn.Sigmoid()
    self.relu = nn.ReLU()
    # self.relu = nn.Tanh()
    # self.relu = nn.LeakyReLU()
    
    self.to(device)

  def forward(self, x):
    x = self.relu(self.input(x))
    x = self.relu(self.layer1(x))
    # x = self.relu(self.layer2(x))
    # x = self.relu(self.layer3(x))
    # x = self.relu(self.layer4(x))
    x = self.output(x)
    
    return x
  
  def train(self, epochs):
    """
    Train the model
    """
    x_train = self.x_train
    x_test = self.x_test
    y_train = self.y_train
    y_test = self.y_test
    
    loss_train_data = np.zeros(epochs) # Store loss data for training
    loss_test_data = np.zeros(epochs) # Store loss data for testing
    
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0005) # Adam optimizer
    loss_function = nn.CrossEntropyLoss() # Cross entropy loss function
    
    for i in range(epochs): # Loop through epochs
      # Train loss
      optimizer.zero_grad()
      y_pred_train = self.forward(x_train)
      train_loss = loss_function(y_pred_train, y_train)
      train_loss.backward()
      
      # Apply weight of 2 to gradients of columns 1 and 2
      with torch.no_grad():
        self.input.weight.grad[:, 0] *= 2
        self.input.weight.grad[:, 1] *= 2
      
      optimizer.step()
      
      # Test loss
      y_pred_test = self.forward(x_test)
      test_loss = loss_function(y_pred_test, y_test)
      
      loss_train_data[i] = train_loss.item()
      loss_test_data[i] = test_loss.item()
      
      if i % 1000 == 0:
        print(f"Epoch {i} Train Loss: {train_loss.item():.4f} Test Loss: {test_loss.item():.4f}")
    
    self.plot(loss_train_data, loss_test_data)

  
  def plot(self, loss_train, loss_test):
    """
    Plots the loss over iterations for training and testing
    """
    plt.figure(figsize=(6, 5))
    plt.plot(loss_train, label='Train Loss')
    plt.plot(loss_test, label='Test Loss')
    plt.legend()
    plt.title('Loss over iterations for training and testing')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    

def plot_decision_boundaries(model, x, y, feature_indices, title):
    """
    Plot decision boundaries for the given model and a pair of features.
    """
    
    # Scale data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Extract the specific feature pair
    x_pair = x_scaled[:, feature_indices]
    
    # Generate a mesh grid over the range of the feature pair
    h = 0.005
    x_min, x_max = x_pair[:, 0].min() - 1, x_pair[:, 0].max() + 1
    y_min, y_max = x_pair[:, 1].min() - 1, x_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Combine mesh grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Match features to grid points
    grid_points_extended = np.zeros((grid_points.shape[0], x.shape[1]))
    grid_points_extended[:, feature_indices[0]] = grid_points[:, 0]
    grid_points_extended[:, feature_indices[1]] = grid_points[:, 1]
    
    # Class for each point
    grid_tensor = torch.FloatTensor(grid_points_extended)
    with torch.no_grad():
      predictions = model(grid_tensor)
    predicted_classes = predictions.argmax(axis=1).numpy()
    
    # Decision boundaries
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, predicted_classes.reshape(xx.shape), alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Original datapts
    plt.scatter(x_pair[:, 0], x_pair[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel(f"Feature {feature_indices[0] + 1} (scaled)")
    plt.ylabel(f"Feature {feature_indices[1] + 1} (scaled)")
    plt.savefig(f"{title}.png")

if __name__ == "__main__":
  # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  device = torch.device("cpu")
  data = Data()
  x, y = data.gen_data("irisdata.csv")
  
  model = Model(x, y, device)
  model.train(15000) 

  plot_decision_boundaries(model, x, y, feature_indices=(0, 1), title="Decision Boundary for Sepal")
  plot_decision_boundaries(model, x, y, feature_indices=(2, 3), title="Decision Boundary for Petal")