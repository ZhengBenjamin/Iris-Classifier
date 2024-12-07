import pandas as pd
import numpy as np

class DataGenerator:

  def __init__(self, k: int, data, two_classes_only=False) -> None:
    self.k = 0
    self.data_vectors = np.empty(0)
    self.classes = np.empty(0)
    self.flowers = []

    self.gen_data_vectors(data, two_classes_only)

  def gen_data_vectors(self, data, two_classes_only=False) -> None:
    """
    Generates data vectors from csv file
    
    :param data: csv file
    :param two_classes_only: if True, only uses classes 2 and 3 from iris dataset
    """
    
    df = pd.read_csv(data)
    df_elements = df.iloc[:, 2:-1]
    df_flowers = df.iloc[:, -1]

    if two_classes_only:
      # Find indices of class 2 and 3
      indices = []
      for i in range(len(df_flowers)):
        if df_flowers.values[i] == "versicolor" or df_flowers[i] == "virginica":
          indices.append(i)
      
      data_vectors = np.empty((len(df_elements.values[0]), len(indices)))

      # Adds row to data vectors 
      for i in range(len(indices)): # Row
        for j in range(len(df_elements.values[0])): # Col
          data_vectors[j][i] = df_elements.values[indices[i]][j]

      self.data_vectors = data_vectors
      self.flowers = df_flowers.values[indices].tolist()

    else:

      data_vectors = np.empty((len(df_elements.values[0]), len(df))) 

      for i in range(len(df_elements)): # For each row in csv 
        for j in range(len(df_elements.values[0])): # For each column in csv 
          data_vectors[j][i] = df_elements.values[i][j]

      self.data_vectors = data_vectors
      self.flowers = df_flowers.values.tolist() 

    # Append flower classes to classes matrix
    self.classes = np.empty(len(self.flowers))
    
    if two_classes_only:
      for i in range(len(self.flowers)):
        if self.flowers[i] == "versicolor":
          self.classes[i] = 0
        else:
          self.classes[i] = 1
    else:
      for i in range(len(self.flowers)):
        if self.flowers[i] == "setosa":
          self.classes[i] = 0
        elif self.flowers[i] == "versicolor":
          self.classes[i] = 1
        else:
          self.classes[i] = 2

if __name__ == '__main__':
  data = DataGenerator(2, 'irisdata.csv', True)
  print(data.flowers)
  print(data.data_vectors)
