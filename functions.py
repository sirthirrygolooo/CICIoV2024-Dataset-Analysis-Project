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
        # plot_confusion_matrix(model, X_test, y_test, f"{model_name} {dataset_name}")
        plot_normalized_confusion_matrix(model,X_test, y_test, f"{model_name} {dataset_name}")
        plot_feature_importance(model, f"{model_name} {dataset_name}", X_test.columns, top_number=5)

        y_predict = model.predict

def prepare_data(df):
    X = df.drop(columns=['isMechant'])
    y = LabelEncoder().fit_transform(df['isMechant'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data(df, target_column='isMechant', test_size=0.2, random_state=42):
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame.")
    
    X = df.drop(columns=[target_column])
    y = LabelEncoder().fit_transform(df[target_column])
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


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
    sns.barplot(x='Model', y='Execution Time', data=exec_df, palette='Set2', hue='Model', legend=False)
    plt.title('Temps d\'exécution des modèles')
    plt.xlabel('Modèle')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.show()

    return exec_df

def aggregate_columns(df, id_column, group_size=100, excluded_columns=None):
    if excluded_columns is None:
        excluded_columns = []
    
    aggregated_data = []

    grouped = df.groupby(id_column)
    
    for id_value, group in grouped:
        for i in range(0, len(group), group_size):
            sub_group = group.iloc[i:i + group_size]
            mean_values = sub_group.drop(columns=excluded_columns).mean(axis=0).round().astype(int)
            for col in excluded_columns:
                if col in sub_group.columns:
                    mean_values[col] = int(sub_group[col].iloc[0])
            mean_values[id_column] = id_value
            aggregated_data.append(mean_values)

    aggregated_df = pd.DataFrame(aggregated_data)
    
    return aggregated_df

def aggregate_columns2(df, id_column, group_size=100, excluded_columns=None):
    if excluded_columns is None:
        excluded_columns = []

    df['unique'] = df.drop(columns=excluded_columns, errors='ignore').nunique(axis=1)
    aggregated_columns = df.T.groupby(df.columns).sum().T

    aggregated_data = []

    grouped = aggregated_columns.groupby(id_column)

    for id_value, group in grouped:
        for i in range(0, len(group), group_size):
            sub_group = group.iloc[i:i + group_size]
            
            mean_values = sub_group.drop(columns=excluded_columns, errors='ignore').mean(axis=0).round().astype(int)
            
            for col in excluded_columns:
                if col in sub_group.columns:
                    mean_values[col] = int(sub_group[col].iloc[0])
            
            mean_values[id_column] = id_value
            
            aggregated_data.append(mean_values)

    # Transformer en DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)

    return aggregated_df

def aggregate_binary_dataframe(df,id_prefix='ID', group_size=519, excluded_columns=None):
    if excluded_columns is None:
        excluded_columns = []
    
    id_columns = [col for col in df.columns if col.startswith(id_prefix)]
    aggregated_data = []
    grouped = df.groupby(id_columns)
    
    for id_values, group in grouped:
        for i in range(0, len(group), group_size):
            sub_group = group.iloc[i:i + group_size]
            
            mean_values = sub_group.drop(columns=excluded_columns + id_columns).mean(axis=0)
            
            for col in excluded_columns:
                if col in sub_group.columns:
                    mean_values[col] = sub_group[col].iloc[0]
            
            for id_col in id_columns:
                mean_values[id_col] = sub_group[id_col].iloc[0]
            
            aggregated_data.append(mean_values)
    
    aggregated_df = pd.DataFrame(aggregated_data)
    
    return aggregated_df

def plot_normalized_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True, 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    
    plt.title(f"Normalized Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.show()

def plot_normalized_confusion_matrix_with_forces(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    n_classes = cm.shape[0]
    FU = 0
    FL = 0
    for i in range(n_classes):
        FU += cm_normalized[i, i+1:].sum()
        FL += cm_normalized[i, :i].sum()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True, 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    
    plt.title(f"Normalized Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    print(f"Force Upper (FU): {FU:.2f}")
    print(f"Force Lower (FL): {FL:.2f}")
    
    return cm_normalized, FU, FL

def diagnostic(models, df):
    X, y = df.drop(columns=['isMechant', 'is_spoofing']), df['isMechant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    total_correct = 0
    total_incorrect = 0
    execution_times = []

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        correct_predictions = np.trace(cm)
        incorrect_predictions = cm.sum() - correct_predictions
        total_correct += correct_predictions
        total_incorrect += incorrect_predictions
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        spoofing_df = df[df['isMechant'] == 1]
        if len(spoofing_df) > 0:
            X_spoof, y_spoof = spoofing_df.drop(columns=['isMechant', 'is_spoofing']), spoofing_df['is_spoofing']
            X_train_spoof, X_test_spoof, y_train_spoof, y_test_spoof = train_test_split(X_spoof, y_spoof, test_size=0.2, random_state=42)

            model_secondary = model
            model_secondary.fit(X_train_spoof, y_train_spoof)
            y_pred_spoof = model_secondary.predict(X_test_spoof)

            cm_spoof = confusion_matrix(y_test_spoof, y_pred_spoof)
            correct_spoof = np.trace(cm_spoof)
            incorrect_spoof = cm_spoof.sum() - correct_spoof

            results.append({
                'Model': model_name,
                'Correct Predictions (Attack)': correct_predictions,
                'Incorrect Predictions (Attack)': incorrect_predictions,
                'Correct Predictions (Spoofing)': correct_spoof,
                'Incorrect Predictions (Spoofing)': incorrect_spoof,
                'Execution Time (s)': execution_time
            })

    df_results = pd.DataFrame(results)

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.bar(df_results['Model'], df_results['Correct Predictions (Attack)'], color='green', label='Correct Predictions')
    plt.bar(df_results['Model'], df_results['Incorrect Predictions (Attack)'], color='red', bottom=df_results['Correct Predictions (Attack)'], label='Incorrect Predictions')
    plt.title('Correct vs Incorrect Predictions for Attacks')
    plt.ylabel('Number of Predictions')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df_results['Model'], df_results['Execution Time (s)'], marker='o', color='blue', label='Execution Time')
    plt.title('Execution Time per Model')
    plt.ylabel('Time (s)')
    plt.xlabel('Model')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Total Correct Predictions: {total_correct}")
    print(f"Total Incorrect Predictions: {total_incorrect}")
    return df_results


