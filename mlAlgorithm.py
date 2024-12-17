# mlAlgorithm.py
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class MLAlgorithms:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column

    def preprocess_data(self):
        print("Preprocessing data...")
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        # Scale features
        scaler = StandardScaler()    #scales data to have a mean of 0 standart deviation of 1

        self.X = scaler.fit_transform(self.X)

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Preprocessing complete.\n")

    def random_forest(self):
        print("Running Random Forest Regression...")

        start_time = time.time()

        # Initialize Random Forest model with parameters
        rf_model = RandomForestRegressor(
            n_estimators=200,      # Number of trees in the forest
            max_depth=10,          # Maximum depth of the trees
            min_samples_split=5,   # Minimum samples required to split a node
            min_samples_leaf=2,    # Minimum samples required at a leaf node
            random_state=42        # seed number
        )

        # Train and predict
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)

        end_time = time.time()
        elapsed_time = end_time - start_time


        # Evaluation metrics
        rf_mse = mean_squared_error(self.y_test, rf_pred)
        rf_r2 = r2_score(self.y_test, rf_pred)

        # Plot results
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, rf_pred, color='green', alpha=0.6, label='Predicted')
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--', label='Ideal Fit')
        plt.title("Random Forest Regression")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.text(0.22, 0.90, f"MSE: {rf_mse:.2f}\nR²: {rf_r2:.2f}", transform=plt.gca().transAxes)
        plt.show()

        print(f"Random Forest -> MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")
        print(f"Training Time: {elapsed_time:.2f} seconds")

    def correlation_matrix(self):
        print("Generating Correlation Matrix...")
        correlation_matrix = self.data.corr()

        # Plot correlation matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Similarity (Correlation Matrix)")
        plt.show()

        #print("Correlation Matrix:\n", correlation_matrix)
