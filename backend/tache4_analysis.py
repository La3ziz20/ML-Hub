import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys

# Add parent dir to path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.data_service import data_service

def main():
    mlflow.set_tracking_uri("sqlite:///./data/mlruns.db")
    mlflow.set_experiment("Task4_Classification")
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = data_service.prepare_data(target_column="Category")
    feature_names = X_train.columns
    
    print("\n--- 1. Feature Importances ---")
    with mlflow.start_run(run_name="RF_Feature_Importance"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, rf.predict(X_test))
        mlflow.log_metric("accuracy", test_acc)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 3 most important features:")
        for i in range(3):
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(10), importances[indices][:10], align="center")
        plt.xticks(range(10), [feature_names[i] for i in indices][:10], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        print("Saved feature_importance.png")
        
    print("\n--- 2. Stabilité des prédictions ---")
    random_states = [42, 123, 456, 789, 999]
    accuracies = []
    for rs in random_states:
        with mlflow.start_run(run_name=f"RF_Stability_rs_{rs}"):
            rf_stab = RandomForestClassifier(n_estimators=50, random_state=rs)
            rf_stab.fit(X_train, y_train)
            acc = accuracy_score(y_test, rf_stab.predict(X_test))
            accuracies.append(acc)
            mlflow.log_param("random_state", rs)
            mlflow.log_metric("accuracy", acc)
    
    print("Accuracies with different random_states:", accuracies)
    print(f"Mean: {np.mean(accuracies):.4f}, Std Dev: {np.std(accuracies):.4f}")
    
    print("\n--- 3. Analyse des erreurs ---")
    y_pred = rf.predict(X_test)
    errors_idx = np.where(y_pred != y_test)[0]
    print(f"Total misclassified: {len(errors_idx)} out of {len(y_test)}")
    
    # Select 3 errors
    for i in range(min(3, len(errors_idx))):
        idx = errors_idx[i]
        real_val = y_test.iloc[idx]
        pred_val = y_pred[idx]
        print(f"\nError {i+1}: True={real_val}, Pred={pred_val}")
        print("Features:")
        # Just print the most important numeric features for context
        top_features = [feature_names[indices[0]], feature_names[indices[1]], feature_names[indices[2]]]
        for f in top_features:
            print(f"  {f}: {X_test.iloc[idx][f]}")
            
    print("\n--- 4. Biais et Variance ---")
    n_estimators_list = [10, 50, 100]
    max_depth_list = [5, 10, None]
    
    results = []
    print(f"{'n_estimators':<12} | {'max_depth':<10} | {'Train Acc':<10} | {'Test Acc':<10} | {'Biais':<8} | {'Variance':<8}")
    print("-" * 70)
    for n_est in n_estimators_list:
        for md in max_depth_list:
            with mlflow.start_run(run_name=f"RF_Grid_n{n_est}_d{md}"):
                rf_grid = RandomForestClassifier(n_estimators=n_est, max_depth=md, random_state=42)
                rf_grid.fit(X_train, y_train)
                
                train_acc = accuracy_score(y_train, rf_grid.predict(X_train))
                test_acc = accuracy_score(y_test, rf_grid.predict(X_test))
                
                bias = 1 - train_acc
                variance = train_acc - test_acc
                
                mlflow.log_params({"n_estimators": n_est, "max_depth": str(md)})
                mlflow.log_metrics({
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "bias": bias,
                    "variance": variance
                })
                
                print(f"{n_est:<12} | {str(md):<10} | {train_acc:<10.4f} | {test_acc:<10.4f} | {bias:<8.4f} | {variance:<8.4f}")
                results.append((n_est, md, train_acc, test_acc, bias, variance))
                
    print("\n--- 5. Comparaison avec Arbre de Décision ---")
    with mlflow.start_run(run_name="Decision_Tree_Baseline"):
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
        dt_test_acc = accuracy_score(y_test, dt.predict(X_test))
        mlflow.log_metrics({"train_acc": dt_train_acc, "test_acc": dt_test_acc})
        
        print(f"Decision Tree - Train Acc: {dt_train_acc:.4f}, Test Acc: {dt_test_acc:.4f}")
        print(f"Random Forest (100 est, None depth) - Train Acc: {results[-1][2]:.4f}, Test Acc: {results[-1][3]:.4f}")

if __name__ == "__main__":
    main()
