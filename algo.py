from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
import dataframe_dec as decimal
import dataframe_bin as binary

X = decimal.df_combined.drop(columns=['isMechant'])
y = decimal.df_combined['isMechant']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Division des données pour entrainement / test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# xgboots
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# lightgbm
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)

# XtraTrees
extra_tree_model = ExtraTreesClassifier()
extra_tree_model.fit(X_train, y_train)

evaluate_model(xgb_model, X_test, y_test, "XGBoost Dec")
X_columns_XG = X.columns
plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost Dec")

evaluate_model(lgbm_model, X_test, y_test, "LightGBM Dec")
X_columns_LGBM = X.columns
plot_confusion_matrix(lgbm_model, X_test, y_test, "LightGBM Dec")

evaluate_model(extra_tree_model, X_test, y_test, "ExtraTrees Dec")
X_columns_ET = X.columns
plot_confusion_matrix(extra_tree_model, X_test, y_test, "EtraTrees Dec")

plot_feature_importance(xgb_model, "XGBoost Dec", X_columns_XG)
plot_feature_importance(lgbm_model, "LightGBM Dec", X_columns_LGBM)
plot_feature_importance(extra_tree_model, "ExtraTrees Dec", X_columns_ET)

# Binary
X = binary.df_combined.drop(columns=['isMechant'])
y = binary.df_combined['isMechant']

y_encoded = le.fit_transform(y)

# Division des données pour entrainement / test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# xgboots
xgb_model.fit(X_train, y_train)

# lightgbm
lgbm_model.fit(X_train, y_train)

# XtraTrees
extra_tree_model.fit(X_train, y_train)

evaluate_model(xgb_model, X_test, y_test, "XGBoost Binary")
X_columns_XG = X.columns
plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost Binary")
y_pred = xgb_model.predict()
get_errors(X_test, y_test, y_pred)

evaluate_model(lgbm_model, X_test, y_test, "LightGBM Binary")
X_columns_LGBM = X.columns
plot_confusion_matrix(lgbm_model, X_test, y_test, "LightGBM Binary")

evaluate_model(extra_tree_model, X_test, y_test, "ExtraTrees Binary")
X_columns_ET = X.columns
plot_confusion_matrix(extra_tree_model, X_test, y_test, "EtraTrees Binary")

plot_feature_importance(xgb_model, "XGBoost Binary", X_columns_XG, 5)
plot_feature_importance(lgbm_model, "LightGBM Binary", X_columns_LGBM, 5)
plot_feature_importance(extra_tree_model, "ExtraTrees Binary", X_columns_ET, 5)