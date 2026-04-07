"""
Model Training for MLflow Project - Bank Marketing
Author: Muhammad Muharram Ash shiddiqie
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import os


def load_data(data_dir="bank_preprocessed"):
    """Load preprocessed bank marketing data"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, n_estimators, max_depth, random_state):
    """Train model with MLflow autolog"""
    
    # Enable MLflow autolog - wajib untuk kriteria basic
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="ci_bank_marketing_model"):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        }
        
        # Train model (autolog will capture params & metrics automatically)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Manual log additional metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.log_metric("n_test_samples", len(X_test))
        mlflow.log_metric("n_features", X_train.shape[1])
        
        # Log model explicitly
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return model, accuracy


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Bank Marketing Model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*50)
    print("Bank Marketing - MLflow CI Training")
    print("="*50)
    
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\nTraining model with MLflow autolog...")
    model, accuracy = train_model(
        X_train, X_test, y_train, y_test,
        args.n_estimators, args.max_depth, args.random_state
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
