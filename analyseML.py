from functions import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
import dataframe_dec as decimal
import dataframe_bin as binary

BIN_PATH = './CICIoV2024/binary/'
DEC_PATH = './CICIoV2024/decimal/'
EXPORT_PATH_TESTS_DEC = './CICIoV2024/tests/decimal/'
EXPORT_PATH_TESTS_BIN = './CICIoV2024/tests/binary/'

# Modèles utilisés
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False,
                              eval_metric='logloss',
                              n_estimators=200,
                              learning_rate=0.1,
                              max_depth=5,
                              colsample_bytree=0.8,
                              subsample=0.8
                              ),
    "LightGBM": LGBMClassifier(n_estimators=200,
                               learning_rate=0.1,
                               max_depth=5,
                               num_leaves=50,
                               colsample_bytree=0.8,
                               subsample=0.8
                               ),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200,
                                       max_depth=20,
                                       min_samples_split=5,
                                       min_samples_leaf=2,
                                       max_features='sqrt'
                                       )
}

# ################################ Decimal ################################
# X_train, X_test, y_train, y_test = prepare_data(decimal.df_combined)
# train_and_evaluate(models, X_train, X_test, y_train, y_test, "Dec")
# ################################ Binary #################################
# X_train, X_test, y_train, y_test = prepare_data(binary.df_combined)
# train_and_evaluate(models, X_train, X_test, y_train, y_test, "Binary")

################################ Analyse temp d'éxecution ############################
# X_train, X_test, y_train, y_test = prepare_data(decimal.df_combined)
# execution_times_df = analyze_execution_time(models, X_train, y_train)

################################ Aggrégation #########################################

# aggregated_df.to_csv(f'{EXPORT_PATH_TESTS_DEC}aggregated_df.csv', index=False)

# X_train,X_test,y_train,y_test = prepare_data(decimal.aggregated_df)
# train_and_evaluate(models, X_train, X_test, y_train, y_test, "Decimal Aggregated")

# X_train,X_test,y_train,y_test = prepare_data(binary.aggregated_df)
# train_and_evaluate(models, X_train, X_test, y_train, y_test, "Binary Aggregated")

X_train, X_test, y_train, y_test = prepare_data(decimal.aggregated_df)
train_and_evaluate(models['XGBoost'], X_train, X_test, y_train, y_test, "Dec aggregated")
plot_normalized_confusion_matrix(model=models['XGBoost'],X_test=X_test,y_test=y_test, model_name="XGBoost")