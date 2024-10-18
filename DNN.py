import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import des dataframes
df_DoS = pd.read_csv('./CICIoV2024/decimal/decimal_DoS.csv')
df_spoofing_Gas = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-GAS.csv')
df_spoofing_RPM = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-RPM.csv')
df_spoofing_speed = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-SPEED.csv')
df_spoofing_steering = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-STEERING_WHEEL.csv')

df_benign = pd.read_csv('./CICIoV2024/decimal/decimal_benign.csv')
df_atk = pd.concat([df_DoS, df_spoofing_Gas, df_spoofing_RPM, df_spoofing_speed, df_spoofing_steering], ignore_index=True)

# Feature selection
df_atk['isMechant'] = 1
df_benign['isMechant'] = 0

# On combine le tout et on enlève les colonnes inutiles
df_combined = pd.concat([df_atk, df_benign], ignore_index=True)
df_combined = df_combined.drop(columns=['ID','label','specific_class', 'category'])

# print(df_combined)

# On crée nos colonnes X et Y 
X = df_combined.drop(columns=['isMechant'])
y = df_combined['isMechant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# print(X_train, y_train)


def randomForest(X_train, X_test, y_train, y_test):
    # Paramètres du modèle RandomForest
    modeleu = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    # Prédictions sur les nouvelles données

    modeleu.fit(X_train, y_train)
    y_pred = modeleu.predict(X_test)

    # Rapport de perf
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(accuracy * 100,4)}%")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion 
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # Importance des features
    importances = modeleu.feature_importances_
    feature_names = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.show()

    # Donées mal classées ~ On regarde ou le test et la prédiction ne correspondaient pas
    misclassified = X_test[y_test != y_pred]
    print("Y'a eu erreur ici :")
    print(misclassified)

def adaBoost(X_train, X_test, y_train, y_test):
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    )

    # Prédictions sur les nouvelles données

    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)

    # Rapport de perf
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(accuracy * 100,4)}%")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion 
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # Importance des features
    importances = ada.feature_importances_
    feature_names = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.show()

    # Donées mal classées ~ On regarde ou le test et la prédiction ne correspondaient pas
    misclassified = X_test[y_test != y_pred]
    print("Y'a eu erreur ici :")
    print(misclassified)


# Construction du modèle DNN
def create_dnn_model():
    model = Sequential()
    
    # Couche d'entrée (8 caractéristiques d'entrée : DATA_0 à DATA_7)
    model.add(Dense(64, input_dim=8, activation='relu'))
    
    # Deuxième couche cachée
    model.add(Dense(32, activation='relu'))
    
    # Troisième couche cachée
    model.add(Dense(16, activation='relu'))
    
    # Couche de sortie (1 neurone pour la classification binaire)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilation du modèle avec l'optimiseur Adam et l'entropie croisée binaire
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def dnn(X_train, X_test, y_train, y_test):
    # Créer le modèle
    neuroneus = create_dnn_model()

    # Entraîner le modèle sur les données d'entraînement
    neuroneus.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    y_pred = neuroneus.predict(X_test)

    # Rapport de perf
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(accuracy * 100,4)}%")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion 
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # Donées mal classées ~ On regarde ou le test et la prédiction ne correspondaient pas
    misclassified = X_test[y_test != y_pred]
    print("Y'a eu erreur ici :")
    print(misclassified)

print("========== AdaBoost ===========")
adaBoost(X_train, X_test, y_train, y_test)
print("===============================")
print("  ")
print("======== RandomForest =========")
randomForest(X_train,X_test,y_train,y_test)
print("===============================")
print("===== DeepNeuralNetwork ======")
dnn(X_train,X_test,y_train,y_test)
print("==============================")