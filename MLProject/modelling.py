"""
Model Training for MLflow Project
Author: Muhammad Muharram Ash shiddiqie
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os


def load_data(data_dir="wine_preprocessed"):
    """Load preprocessed data"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, n_estimators, max_depth, random_state):
    """Train model with MLflow logging"""
    
    # Enable MLflow autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="ci_rf_model"):
        # Log parameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        }
        
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        
        return model, accuracy


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Wine Classification Model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\nTraining model with MLflow...")
    model, accuracy = train_model(
        X_train, X_test, y_train, y_test,
        args.n_estimators, args.max_depth, args.random_state
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
