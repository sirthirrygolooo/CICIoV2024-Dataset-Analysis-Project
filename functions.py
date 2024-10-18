import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import pandas as pd
import time

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))
    print("\n")

def train_and_evaluate(models, X_train, X_test, y_train, y_test, dataset_name):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, f"{model_name} {dataset_name}")
        plot_confusion_matrix(model, X_test, y_test, f"{model_name} {dataset_name}")
        plot_feature_importance(model, f"{model_name} {dataset_name}", X_test.columns, top_number=5)

def prepare_data(df):
    X = df.drop(columns=['isMechant'])
    y = LabelEncoder().fit_transform(df['isMechant'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_feature_importance(model, model_name, feature_names, top_number=None):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    if top_number is not None:
        indices = indices[:top_number]
    
    top_importance = importance[indices]
    top_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance - Top {len(top_features)} Features - {model_name}")
    plt.bar(range(len(top_features)), top_importance, align="center")
    plt.xticks(range(len(top_features)), top_features, rotation=90)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['BENIGN', 'ATTACK'], 
                yticklabels=['BENIGN', 'ATTACK'])
    plt.title(f"Matrice de confusion - {model_name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.tight_layout()
    plt.show()

def get_errors(X_test, y_test, y_pred):
    misclassified = X_test[y_test != y_pred]
    print("Erreurs :")
    print(misclassified)

def analyze_execution_time(models, X_train, y_train):
    execution_times = []

    for model_name, model in models.items():
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        execution_times.append((model_name, execution_time))
        
        print(f"{model_name} a pris {execution_time:.4f} secondes")


    exec_df = pd.DataFrame(execution_times, columns=['Model', 'Execution Time'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Execution Time', data=exec_df, palette='Set2')
    plt.title('Temps d\'exécution des modèles')
    plt.xlabel('Modèle')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.show()

    return exec_df
