# Neural Network for Iris Dataset Classification

This Python script implements a neural network using PyTorch to classify the Iris dataset. The implementation includes data preprocessing, model definition, training, and visualization of decision boundaries. **Average accuracy of 0.97.**

## Overview

- **Data Preprocessing**: 
  - The dataset is loaded and species labels are mapped to numerical values (`setosa: 0, versicolor: 1, virginica: 2`).
  - Features are scaled using `StandardScaler` for normalization.
  - Data is split into training (70%) and testing (30%) sets.

- **Model Architecture**:
  - The model consists of:
    - Input layer matching the feature size.
    - Hidden layers with ReLU activation.
    - Output layer with three nodes for the three classes.
  - Alternative configurations can include additional layers or different activation functions (e.g., Sigmoid, Tanh, Leaky ReLU).

- **Training**:
  - Optimized using Adam with a learning rate of 0.0005.
  - Uses CrossEntropyLoss for multi-class classification.
  - Implements a gradient adjustment to apply extra weight to specific input features.
  - Logs and plots loss values for both training and testing phases over specified epochs.

- **Visualization**:
  - Decision boundaries are plotted for specific feature pairs using a mesh grid. 
  - The boundaries show the regions predicted for each class, overlaid with the actual data points.

## Example Outputs:

![Decision Boundary for Sepal](https://github.com/user-attachments/assets/61050b1e-be3b-40f5-9132-54e4f2251fe5)
![Decision Boundary for Petal](https://github.com/user-attachments/assets/a841c1af-0ad6-4b97-b225-704849a84ba1)
![loss](https://github.com/user-attachments/assets/2e27602e-7d2a-4dd2-825b-0839383ce3c2)

## Key Classes and Functions

### `Data` Class
Handles data preprocessing tasks, including:
- Reading and mapping labels in the Iris dataset.
- Splitting the dataset into input features (`x`) and output labels (`y`).

### `Model` Class
Defines the neural network, including:
- Constructor (`__init__`): Sets up layers, activation functions, and data preprocessing for training/testing sets.
- `forward`: Propagates input through the network.
- `train`: Trains the model, adjusts gradients, and plots training/testing loss.
- `plot`: Visualizes loss over epochs.

### `plot_decision_boundaries`
Visualizes decision boundaries for feature pairs:
- Generates a mesh grid for the selected features.
- Maps predictions from the model to grid points and visualizes them.
- Plots actual data points for reference.

## Usage
1. Ensure the Iris dataset is available as `irisdata.csv`.
2. Run the script; the following operations are performed:
   - Model training for 15,000 epochs.
   - Loss visualization saved as `loss.png`.
   - Decision boundary plots for:
     - Sepal features (`Feature 1` vs. `Feature 2`).
     - Petal features (`Feature 3` vs. `Feature 4`).
   - Decision boundary plots are saved as `Decision Boundary for Sepal.png` and `Decision Boundary for Petal.png`.

