# **Iris Classifier**

# **Overview**

The Iris Classifier is a Python program that implements a single-layer neural network to classify the Iris dataset into two distinct classes based on petal length and width. It uses sigmoid activation and gradient descent for weight optimization, visualizing the decision boundaries and error reduction throughout training.

# **Features**

* Binary Classification: Focuses on differentiating two classes of the Iris dataset (e.g., Versicolor and Virginica).
* Dynamic Decision Boundaries: Plots boundaries during and after training, showcasing the network's progress.
* Customizable Training Parameters: Allows adjustments to step size, iterations, and convergence tolerance.
* Visualization Tools: Includes scatter plots of the dataset and learning curves to monitor performance.
* Error Metrics: Calculates Mean Squared Error (MSE) to evaluate the model's performance over time.

# **How it works**

**1. Data Preprocessing**
The program preprocesses the Iris dataset, converting petal length and width into feature vectors. Classes are mapped to binary labels for simplified training.

**2. Network Initialization**
The single-layer neural network starts with random weights and a bias term.

**3. Training**
Using gradient descent, the network adjusts weights to minimize MSE iteratively. The program stops early if error improvements fall below a predefined tolerance.

**4. Visualization**
Plots include:
- Decision boundaries separating the two classes.
- Learning curves showing error reduction over training iterations.

# **Results**
The trained network effectively classifies data points and produces visually interpretable decision boundaries

Visualization of training process:

![Iris Data Initial](https://github.com/user-attachments/assets/645f0278-2da5-459a-8a0b-372ea9549de8)
![Iris Data Intermediate](https://github.com/user-attachments/assets/784f3828-a6fa-41ba-94ca-7f8435f262ec) 
![Iris Data Trained](https://github.com/user-attachments/assets/66a5a221-d1e3-4fa5-8898-de15c962fe8b)

Mean square error as a function of iterations:

![Learning Curve Final](https://github.com/user-attachments/assets/6f8c276e-68fe-45eb-b86a-0f57b7e4066c)

# **Dependencies**

* Python 3.x
* NumPy
* Matplotlib
* Pandas

