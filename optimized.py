from functions import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
import dataframe_dec as decimal
import dataframe_bin as binary

def train_and_evaluate_with_tuning(models, param_grids, X_train, X_test, y_train, y_test, dataset_name):
    for model_name, model in models.items():
        print(f"Recherche des meilleurs paramètres pour {model_name}...")
        
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Meilleurs paramètres pour {model_name} : {grid_search.best_params_}")
        
        evaluate_model(best_model, X_test, y_test, f"{model_name} {dataset_name}")
        plot_confusion_matrix(best_model, X_test, y_test, f"{model_name} {dataset_name}")
        plot_feature_importance(best_model, f"{model_name} {dataset_name}", X_test.columns, top_number=5)

def prepare_data(df):
    X = df.drop(columns=['isMechant'])
    y = LabelEncoder().fit_transform(df['isMechant'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Params
param_grids = {
    "XGBoost": {
        'use_label_encoder':[False],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.6, 0.8, 1],
        'subsample': [0.6, 0.8, 1]
    },
    "LightGBM": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [-1, 5, 10],
        'num_leaves': [31, 50, 100],
        'colsample_bytree': [0.6, 0.8, 1],
        'subsample': [0.6, 0.8, 1]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
}

# Models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "ExtraTrees": ExtraTreesClassifier()
}

X_train, X_test, y_train, y_test = prepare_data(decimal.df_combined)
train_and_evaluate_with_tuning(models, param_grids, X_train, X_test, y_train, y_test, "Dec")

X_train, X_test, y_train, y_test = prepare_data(binary.df_combined)
train_and_evaluate_with_tuning(models, param_grids, X_train, X_test, y_train, y_test, "Binary")