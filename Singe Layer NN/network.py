import numpy as np
import sys
import math
import random

from data_generator import DataGenerator as data

"""
This file contains the Network class which is responsible for training the neural network
"""

class Network: 
  
  def __init__(self) -> None:
    self.data = data(2, 'irisdata.csv', True)
    self.data_vectors = np.vstack([np.ones(len(self.data.data_vectors[0])), self.data.data_vectors])
    self.error_over_time = []
    self.weights_over_time = [] 

    self.weights = self.random_weights()


  def random_weights(self) -> np.ndarray:
    """
    Random weights are assigned based on min max values of x and y intercepts
    Guarantees that intiial weights are able to be plotted
    """
    x_intercept = np.random.uniform(2.5, 7.5)
    y_intercept = np.random.uniform(0.5, 3)

    w1 = random.uniform(-2, 2)
    w2 = (3 - 0.5) / (7.5 - 2.5)
    w0 = -w1 * x_intercept - w2 * y_intercept

    return np.array([w0, w1, w2])


  def output(self, data, weights) -> np.ndarray:
    """
    Computes the output of simple one layer neural network using sigmoid non linearity

    :param data: data vectors
    :param weights: weights
    """
    sum = np.matmul(weights, data)
    # print(f"Data: {data}, Weights: {weights}, Sum: {sum}")

    return 1 / (1 + np.exp(-sum))
  
  def train(self, step_size=0.005, iterations=10000, tolerance=1e-6):
    """
    Trains the neural network
    
    :param step_size: step size
    :param iterations: number of iterations
    :param tolerance: tolerance, stops training if error is less than tolerance
    """

    for i in range(iterations):
      new_weights = self.weights - (step_size * self.sum_gradient(self.data_vectors, self.weights))
      self.weights = new_weights
      self.error_over_time.append(self.mean_squared_error(self.data_vectors, self.weights, self.data.classes))
      self.weights_over_time.append(self.weights)

      if i > 0 and abs(self.error_over_time[-1] - self.error_over_time[-2]) < tolerance:
        print(f"Stopping early at iteration {i}, MSE: {self.error_over_time[-1]}")
        break
  

    print(f"Final weights: {self.weights}")
  

  def mean_squared_error(self, data, weights, pattern) -> float:
    """
    Computes the mean squared error of the neural network

    :param data: data vectors
    :param weights: weights
    :param basis: basis
    :param pattern: pattern
    """
    network_result = self.output(data, weights) # Network result
    # print(f"Network result: {network_result}")
    # Calculate mean squared error 
    mean_error = sum((pattern - network_result) ** 2) / len(network_result)
    return mean_error 
  
  def sum_gradient(self, data, weight):
    """
    Computes the sum of the gradient

    :param data: data vectors
    :param weight: weight
    """

    sum_gradient = np.zeros(3)

    for i in range(len(data[0])):

      data_point = data[:, i]

      if self.data.flowers[i] == "versicolor":
        actual = 0
      else:
        actual = 1

      sigmoid = 1 / (1 + np.exp(-np.dot(weight, data_point)))

      for j in range(len(sum_gradient)):
        sum_gradient[j] += 2 * (sigmoid - actual) * ((np.exp(-np.dot(weight, data_point))) / (1 + np.exp(-np.dot(weight, data_point))) ** 2) * data_point[j]

    return sum_gradient
