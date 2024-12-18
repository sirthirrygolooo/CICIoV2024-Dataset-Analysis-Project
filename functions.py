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

def aggregate_columns2(df, id_column, group_size=1, excluded_columns=None):
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

def diagnosticv1(models, df):
    X, y = df.drop(columns=['isMechant', 'is_spoofing']), df['isMechant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    total_correct_attack = 0
    total_incorrect_attack = 0
    total_correct_spoof = 0
    total_incorrect_spoof = 0
    execution_times = []

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm_attack = confusion_matrix(y_test, y_pred)
        correct_predictions_attack = np.trace(cm_attack)
        incorrect_predictions_attack = cm_attack.sum() - correct_predictions_attack
        total_correct_attack += correct_predictions_attack
        total_incorrect_attack += incorrect_predictions_attack
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
            correct_predictions_spoof = np.trace(cm_spoof)
            incorrect_predictions_spoof = cm_spoof.sum() - correct_predictions_spoof
            total_correct_spoof += correct_predictions_spoof
            total_incorrect_spoof += incorrect_predictions_spoof

            results.append({
                'Model': model_name,
                'Correct Predictions (Attack)': correct_predictions_attack,
                'Incorrect Predictions (Attack)': incorrect_predictions_attack,
                'Correct Predictions (Spoofing)': correct_predictions_spoof,
                'Incorrect Predictions (Spoofing)': incorrect_predictions_spoof,
                'Execution Time (s)': execution_time
            })

    df_results = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    total_cm_attack = np.array([[total_correct_attack, total_incorrect_attack],
                                [total_incorrect_attack, total_correct_attack]])
    sns.heatmap(total_cm_attack, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix: Attack Detection')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    total_cm_spoof = np.array([[total_correct_spoof, total_incorrect_spoof],
                               [total_incorrect_spoof, total_correct_spoof]])
    sns.heatmap(total_cm_spoof, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix: Spoofing Detection')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')

    width = 0.35
    x = np.arange(len(models))
    axes[1, 0].bar(x - width / 2, df_results['Correct Predictions (Attack)'], width, label='Correct Attack', color='green')
    axes[1, 0].bar(x - width / 2, df_results['Incorrect Predictions (Attack)'], width, bottom=df_results['Correct Predictions (Attack)'], color='red')

    axes[1, 0].bar(x + width / 2, df_results['Correct Predictions (Spoofing)'], width, label='Correct Spoofing', color='blue')
    axes[1, 0].bar(x + width / 2, df_results['Incorrect Predictions (Spoofing)'], width, bottom=df_results['Correct Predictions (Spoofing)'], color='orange')

    axes[1, 0].set_title('Correct and Incorrect Predictions per Model')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df_results['Model'])
    axes[1, 0].legend()

    sns.boxplot(data=execution_times, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Execution Time Distribution')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_xticklabels(["Execution Time"])

    plt.tight_layout()
    plt.show()

    print("\n--- Results Summary ---")
    print(df_results)
    print(f"\nTotal Correct Predictions (Attack): {total_correct_attack}")
    print(f"Total Incorrect Predictions (Attack): {total_incorrect_attack}")
    print(f"Total Correct Predictions (Spoofing): {total_correct_spoof}")
    print(f"Total Incorrect Predictions (Spoofing): {total_incorrect_spoof}")

    return df_results

def diagnosticv2(models, df):
    X, y = df.drop(columns=['isMechant', 'is_spoofing']), df['isMechant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    total_correct_dos = 0
    total_incorrect_dos = 0
    total_correct_spoof = 0
    total_incorrect_spoof = 0
    execution_times = []

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm_attack = confusion_matrix(y_test, y_pred)
        attack_correct = np.trace(cm_attack)
        attack_incorrect = cm_attack.sum() - attack_correct
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

            correct_dos = attack_correct - correct_spoof
            incorrect_dos = attack_incorrect - incorrect_spoof

            total_correct_dos += correct_dos
            total_incorrect_dos += incorrect_dos
            total_correct_spoof += correct_spoof
            total_incorrect_spoof += incorrect_spoof

            results.append({
                'Model': model_name,
                'Correct Predictions (DoS)': correct_dos,
                'Incorrect Predictions (DoS)': incorrect_dos,
                'Correct Predictions (Spoofing)': correct_spoof,
                'Incorrect Predictions (Spoofing)': incorrect_spoof,
                'Execution Time (s)': execution_time
            })

    df_results = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))


    total_cm_attack = np.array([[total_correct_dos, total_incorrect_dos],
                                [total_incorrect_dos, total_correct_dos]])
    sns.heatmap(total_cm_attack, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix: DoS Detection')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    total_cm_spoof = np.array([[total_correct_spoof, total_incorrect_spoof],
                               [total_incorrect_spoof, total_correct_spoof]])
    sns.heatmap(total_cm_spoof, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix: Spoofing Detection')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')

    width = 0.35
    x = np.arange(len(models))
    axes[1, 0].bar(x - width / 2, df_results['Correct Predictions (DoS)'], width, label='Correct DoS', color='green')
    axes[1, 0].bar(x - width / 2, df_results['Incorrect Predictions (DoS)'], width, bottom=df_results['Correct Predictions (DoS)'], color='red')

    axes[1, 0].bar(x + width / 2, df_results['Correct Predictions (Spoofing)'], width, label='Correct Spoofing', color='blue')
    axes[1, 0].bar(x + width / 2, df_results['Incorrect Predictions (Spoofing)'], width, bottom=df_results['Correct Predictions (Spoofing)'], color='orange')

    axes[1, 0].set_title('Correct and Incorrect Predictions per Model')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df_results['Model'])
    axes[1, 0].legend()

    sns.boxplot(data=execution_times, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Execution Time Distribution')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_xticklabels(["Execution Time"])

    plt.tight_layout()
    plt.show()

    print("\n--- Results Summary ---")
    print(df_results)
    print(f"\nTotal Correct Predictions (DoS): {total_correct_dos}")
    print(f"Total Incorrect Predictions (DoS): {total_incorrect_dos}")
    print(f"Total Correct Predictions (Spoofing): {total_correct_spoof}")
    print(f"Total Incorrect Predictions (Spoofing): {total_incorrect_spoof}")

    return df_results

def diagnosticv3(models, df):
    X, y = df.drop(columns=['isMechant', 'is_spoofing', 'spoofing_type']), df['isMechant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    total_correct_dos = 0
    total_incorrect_dos = 0
    total_correct_spoof = 0
    total_incorrect_spoof = 0
    total_correct_types = 0
    total_incorrect_types = 0
    execution_times = []

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm_attack = confusion_matrix(y_test, y_pred)
        attack_correct = np.trace(cm_attack)
        attack_incorrect = cm_attack.sum() - attack_correct
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        spoofing_df = df[df['isMechant'] == 1]
        if len(spoofing_df) > 0:
            X_spoof, y_spoof = spoofing_df.drop(columns=['isMechant', 'is_spoofing', 'spoofing_type']), spoofing_df['is_spoofing']
            X_train_spoof, X_test_spoof, y_train_spoof, y_test_spoof = train_test_split(X_spoof, y_spoof, test_size=0.2, random_state=42)

            model_secondary = model
            model_secondary.fit(X_train_spoof, y_train_spoof)
            y_pred_spoof = model_secondary.predict(X_test_spoof)

            cm_spoof = confusion_matrix(y_test_spoof, y_pred_spoof)
            correct_spoof = np.trace(cm_spoof)
            incorrect_spoof = cm_spoof.sum() - correct_spoof

            spoof_type_df = spoofing_df[spoofing_df['is_spoofing'] == 1]
            if len(spoof_type_df) > 0:
                X_type, y_type = spoof_type_df.drop(columns=['isMechant', 'is_spoofing', 'spoofing_type']), spoof_type_df['spoofing_type']
                X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, random_state=42)

                model_tertiary = model
                model_tertiary.fit(X_train_type, y_train_type)
                y_pred_type = model_tertiary.predict(X_test_type)

                cm_type = confusion_matrix(y_test_type, y_pred_type)
                correct_types = np.trace(cm_type)
                incorrect_types = cm_type.sum() - correct_types
            else:
                correct_types = 0
                incorrect_types = 0

            correct_dos = attack_correct - correct_spoof
            incorrect_dos = attack_incorrect - incorrect_spoof

            total_correct_dos += correct_dos
            total_incorrect_dos += incorrect_dos
            total_correct_spoof += correct_spoof
            total_incorrect_spoof += incorrect_spoof
            total_correct_types += correct_types
            total_incorrect_types += incorrect_types

            results.append({
                'Model': model_name,
                'Correct Predictions (DoS)': correct_dos,
                'Incorrect Predictions (DoS)': incorrect_dos,
                'Correct Predictions (Spoofing)': correct_spoof,
                'Incorrect Predictions (Spoofing)': incorrect_spoof,
                'Correct Predictions (Spoofing Type)': correct_types,
                'Incorrect Predictions (Spoofing Type)': incorrect_types,
                'Execution Time (s)': execution_time
            })

    df_results = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    total_cm_attack = np.array([[total_correct_dos, total_incorrect_dos],
                                [total_incorrect_dos, total_correct_dos]])
    sns.heatmap(total_cm_attack, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix: DoS Detection')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    total_cm_spoof = np.array([[total_correct_spoof, total_incorrect_spoof],
                               [total_incorrect_spoof, total_correct_spoof]])
    sns.heatmap(total_cm_spoof, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix: Spoofing Detection')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')

    total_cm_type = np.array([[total_correct_types, total_incorrect_types],
                              [total_incorrect_types, total_correct_types]])
    sns.heatmap(total_cm_type, annot=True, fmt='d', cmap='Purples', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix: Spoofing Type Classification')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    sns.boxplot(data=execution_times, ax=axes[1, 1], color='blue')
    axes[1, 1].set_title('Execution Time Distribution')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_xticklabels(["Execution Time"])

    plt.tight_layout()
    plt.show()


    print("\n--- Results Summary ---")
    print(df_results)
    print(f"\nTotal Correct Predictions (DoS): {total_correct_dos}")
    print(f"Total Incorrect Predictions (DoS): {total_incorrect_dos}")
    print(f"Total Correct Predictions (Spoofing): {total_correct_spoof}")
    print(f"Total Incorrect Predictions (Spoofing): {total_incorrect_spoof}")
    print(f"Total Correct Predictions (Spoofing Type): {total_correct_types}")
    print(f"Total Incorrect Predictions (Spoofing Type): {total_incorrect_types}")

    return df_results

def diagnostic_final(models, df):

    X, y = df.drop(columns=['isMechant', 'is_spoofing', 'spoofing_type']), df['isMechant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    execution_times = []

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm_attack = confusion_matrix(y_test, y_pred)
        accuracy_attack = accuracy_score(y_test, y_pred)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        spoofing_df = df[df['isMechant'] == 1]
        if len(spoofing_df) > 0:
            X_spoof, y_spoof = spoofing_df.drop(columns=['isMechant', 'is_spoofing', 'spoofing_type']), spoofing_df['is_spoofing']
            X_train_spoof, X_test_spoof, y_train_spoof, y_test_spoof = train_test_split(X_spoof, y_spoof, test_size=0.2, random_state=42)

            model_secondary = model
            model_secondary.fit(X_train_spoof, y_train_spoof)
            y_pred_spoof = model_secondary.predict(X_test_spoof)

            cm_spoof = confusion_matrix(y_test_spoof, y_pred_spoof)
            accuracy_spoof = accuracy_score(y_test_spoof, y_pred_spoof)
        else:
            cm_spoof = None
            accuracy_spoof = None

        spoof_type_df = spoofing_df[spoofing_df['is_spoofing'] == 1]
        if len(spoof_type_df) > 0:
            X_type, y_type = spoof_type_df.drop(columns=['isMechant', 'is_spoofing', 'spoofing_type']), spoof_type_df['spoofing_type']
            X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, random_state=42)

            model_tertiary = model
            model_tertiary.fit(X_train_type, y_train_type)
            y_pred_type = model_tertiary.predict(X_test_type)

            cm_type = confusion_matrix(y_test_type, y_pred_type)
            accuracy_type = accuracy_score(y_test_type, y_pred_type)
        else:
            cm_type = None
            accuracy_type = None

        results.append({
            'Model': model_name,
            'Attack Accuracy': accuracy_attack,
            'Spoofing Accuracy': accuracy_spoof if accuracy_spoof is not None else "N/A",
            'Spoofing Type Accuracy': accuracy_type if accuracy_type is not None else "N/A",
            'Execution Time (s)': execution_time,
            'Confusion Matrix (Attack)': cm_attack,
            'Confusion Matrix (Spoofing)': cm_spoof,
            'Confusion Matrix (Type)': cm_type
        })

    results_df = pd.DataFrame(results)

    print("\n" + "="*30)
    print("       MODEL PERFORMANCE SUMMARY       ")
    print("="*30)
    for idx, row in results_df.iterrows():
        print(f"\n### Model: {row['Model']} ###")
        print(f"- Attack Detection Accuracy: {row['Attack Accuracy']:.2%}")
        print("  Confusion Matrix (DoS Detection):")
        print(row['Confusion Matrix (Attack)'])

        if row['Spoofing Accuracy'] != "N/A":
            print(f"\n- Spoofing Detection Accuracy: {row['Spoofing Accuracy']:.2%}")
            print("  Confusion Matrix (Spoofing Detection):")
            print(row['Confusion Matrix (Spoofing)'])

        if row['Spoofing Type Accuracy'] != "N/A":
            print(f"\n- Spoofing Type Classification Accuracy: {row['Spoofing Type Accuracy']:.2%}")
            print("  Confusion Matrix (Spoofing Type):")
            print(row['Confusion Matrix (Type)'])

        print(f"- Execution Time: {row['Execution Time (s)']:.2f} seconds")
        print("-" * 30)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    results_df.plot(kind='bar', x='Model', y=['Attack Accuracy', 'Spoofing Accuracy', 'Spoofing Type Accuracy'],
                    ax=ax[0], color=['green', 'orange', 'purple'], rot=45)
    ax[0].set_title("Model Accuracies for Detection Tasks")
    ax[0].set_ylabel("Accuracy")

    sns.boxplot(data=results_df['Execution Time (s)'], ax=ax[1], color='blue')
    ax[1].set_title("Model Execution Time Distribution")
    ax[1].set_ylabel("Execution Time (s)")
    ax[1].set_xticklabels(["Execution Time"])

    plt.tight_layout()
    plt.show()

    return results_df

