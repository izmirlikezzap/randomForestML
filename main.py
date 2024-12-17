from mlAlgorithm import MLAlgorithms
import pandas as pd

# Load the dataset
file_path = "hw4_data.xlsx"
data = pd.read_excel(file_path)

# Define target column
target_column = "Temperature"

# Initialize and preprocess
ml = MLAlgorithms(data, target_column)
ml.preprocess_data()

# Run Random Forest
ml.random_forest()

# Generate Correlation Matrix
ml.correlation_matrix()
