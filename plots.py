import matplotlib.pyplot as plt
import numpy as np
import os
import math

from data_generator import DataGenerator as data
from network import Network as network 

class Output: 

  def __init__(self, k: int, two_classes_only=True) -> None:
    self.data = data(k, 'irisdata.csv', two_classes_only) # Change the number of clusters / classes 
    self.net = network()
    self.data_vectors = np.vstack([np.ones(len(self.data.data_vectors[0])), self.data.data_vectors])

    # Plots the 2nd and 3rd class of iris dataset 
    # self.plot_points()

    # # Choose boundary that roughly seperates two classes
    self.plot_boundaries(np.array([-5.7, 0.5, 2]), title="Iris Data Guess")

    # Compute MSE for two different settings of weights
    weights = np.array([-4, 1, 0.5])
    mse = self.net.mean_squared_error(self.data_vectors, weights, self.data.classes)
    self.plot_boundaries(weights, "Iris Data Large MSE")
    print(f"Mean Squared Error 1: {mse}")

    weights = np.array([-5.7, 0.5, 2])
    mse = self.net.mean_squared_error(self.data_vectors, weights, self.data.classes)
    print(f"Mean Squared Error 2: {mse}")
    self.plot_boundaries(weights, "Iris Data Small MSE")

    # Decision boundary changes for small step in weight and bias values
    self.plot_boundaries(weights = np.array([-5, 0.5, 2]), title="Iris Data Original") # Original
    self.plot_boundaries(weights = np.array([-5.018, 0.427, 1.9778]), title="Iris Data Small Step") # Small step: weights - sum_gradient * 0.05

    # Trains the network and plots the progress over time
    self.net.train()
    self.plot_over_time()

  def plot_points(self) -> None:
    """
    Plots the data points
    """
    x, y, color = self.get_petal_dim()
    plt.scatter(x, y, c=color)
    plt.title("Iris Data")
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.show()

  def plot_boundaries(self, weights, title="Iris Data") -> None:
    """
    Plots the data and the decision boundary
    
    Optional Parameters:
    bias: Custom bias for boundaries float
    weights: Custom weights for boundaries list
    """
    plt.clf()

    # If no bias or weights are provided, use the default ones
    if weights is None:
      weights = self.net.weights

    bias = weights[0]

    x, y, color = self.get_petal_dim()

    plt.scatter(x, y, c=color)

    x = np.linspace(2.5, 7.5, 100)
    y = - (weights[1] * np.array(x) + bias) / weights[2]

    plt.fill_between(x, y, 0.5, color='Green', alpha=0.1)
    plt.fill_between(x, y, 3, color='Blue', alpha=0.1)

    plt.plot(x, y)
    plt.xlim(2.5, 7.5)
    plt.ylim(0.5, 3)
    plt.title(title)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')

    script_dir = os.path.dirname(__file__)
    res_dir = os.path.join(script_dir, 'outputs/')

    if not os.path.isdir(res_dir):
      os.makedirs(res_dir)

    plt.savefig(res_dir + title)

  def plot_over_time(self) -> None:
    """
    Plots the progress of the clusters over time
    """
    errors = self.net.error_over_time
    weights = self.net.weights_over_time
    
    self.plot_boundaries(weights[0], title="Iris Data Initial")
    print(weights[0])

    self.plot_boundaries(weights[math.floor(len(weights) / 6)], title="Iris Data Intermediate")
    self.plot_learning_curve(errors[:math.floor(len(errors) / 6)], title="Learning Curve Intermediate")

    self.plot_boundaries(weights[-1], title="Iris Data Trained")
    self.plot_learning_curve(errors, title="Learning Curve Final")


  def plot_learning_curve(self, error_over_time, title="Learning Curve") -> None: 

    plt.clf()

    x = [i for i in range(len(error_over_time))]
    y = error_over_time

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    
    script_dir = os.path.dirname(__file__)
    res_dir = os.path.join(script_dir, 'outputs/')

    if not os.path.isdir(res_dir):
      os.makedirs(res_dir)

    plt.savefig(res_dir + title)


  def get_petal_dim(self) -> tuple:
    values = self.data.data_vectors
    x = []
    y = [] 
    color = []

    for i in range(len(values[0])):
      if self.data.flowers[i] == "versicolor" or self.data.flowers[i] == "virginica":
        x.append(values[0][i])
        y.append(values[1][i])
        color.append("green" if self.data.flowers[i] == "versicolor" else "blue")

    return x, y, color
      

if __name__ == "__main__":
  Output(2)

    