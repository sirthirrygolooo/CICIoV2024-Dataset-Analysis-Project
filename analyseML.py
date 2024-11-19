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
# models = {
#     "XGBoost": XGBClassifier(use_label_encoder=False,
#                               eval_metric='logloss',
#                               n_estimators=200,
#                               learning_rate=0.1,
#                               max_depth=5,
#                               colsample_bytree=0.8,
#                               subsample=0.8
#                               ),
#     "LightGBM": LGBMClassifier(n_estimators=200,
#                                learning_rate=0.1,
#                                max_depth=5,
#                                num_leaves=50,
#                                colsample_bytree=0.8,
#                                subsample=0.8
#                                ),
#     "ExtraTrees": ExtraTreesClassifier(n_estimators=200,
#                                        max_depth=20,
#                                        min_samples_split=5,
#                                        min_samples_leaf=2,
#                                        max_features='sqrt'
#                                        )
# }

models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(min_split_gain=-2),
    # "ExtraTrees": ExtraTreesClassifier()
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

# X_train,X_test,y_train,y_test = prepare_data(decimal.df_combined)
# train_and_evaluate(models, X_train, X_test, y_train, y_test, "Decimal Aggregated")


# analyze_execution_time(models, X_train, y_train)

# X_train,X_test,y_train,y_test = prepare_data(binary.aggregated_df)
# train_and_evaluate(models, X_train, X_test, y_train, y_test, "Binary Aggregated")

# X_train, X_test, y_train, y_test = prepare_data(decimal.aggregated_df)
# train_and_evaluate(models['XGBoost'], X_train, X_test, y_train, y_test, "Dec aggregated")
# print('ok')

# Test FU / FL


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def analyse_time_and_res():
    i = 100
    execution_times = []

    models = {
        "XGBoost": XGBClassifier(eval_metric='logloss'),
    }

    FU_values = []
    FL_values = []

    while i < 300:

        aggregated_df_atk = aggregate_columns(decimal.df_atk_clean, id_column='ID', group_size=i)
        aggregated_df_benign = aggregate_columns(decimal.df_benign_clean, id_column='ID', group_size=i)

        aggregated_df = pd.concat([aggregated_df_atk, aggregated_df_benign], ignore_index=True)
        aggregated_df = aggregated_df.drop(columns=['ID'])

        X_train, X_test, y_train, y_test = prepare_data(aggregated_df)

        misclassification_occurred = False
        misclassification_true_negativ = 0
        misclassification_false_positiv = 0

        FU = 0
        FL = 0

        start_time = time.time()

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # CM
            cm = confusion_matrix(y_test, predictions)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # La c'est les Forces UPPER/LOWER
            n_classes = cm_normalized.shape[0]
            for idx in range(n_classes):
                FU += cm_normalized[idx, idx+1:].sum()
                FL += cm_normalized[idx, :idx].sum()

            for true_label, pred_label in zip(y_test, predictions):
                if (true_label == 1 and pred_label == 0):  
                    misclassification_occurred = True
                    misclassification_true_negativ += 1

                if true_label == 0 and pred_label == 1:
                    misclassification_occurred = True
                    misclassification_false_positiv += 1

        execution_time = time.time() - start_time
        execution_times.append((i, execution_time, not misclassification_occurred, misclassification_true_negativ, misclassification_false_positiv))
        FU_values.append(FU)
        FL_values.append(FL)

        i += 10

    execution_times_df = pd.DataFrame(execution_times, columns=['Group Size (i)', 'Execution Time (s)', 'No Misclassification', 'True Negative', 'False Positive'])

    execution_times_df['Total Misclassifications'] = (
        execution_times_df['True Negative'] + execution_times_df['False Positive']
    )
    execution_times_df['Force Upper (FU)'] = FU_values
    execution_times_df['Force Lower (FL)'] = FL_values

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.grid(visible=True)
    plt.plot(
        execution_times_df['Group Size (i)'], 
        execution_times_df['Total Misclassifications'], 
        marker='x', label='Total Misclassifications', color='blue'
    )
    plt.xlabel('Group Size (i)')
    plt.ylabel('Number of Misclassifications')
    plt.title('Number of Misclassifications by Group Size')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.grid(visible=True)
    plt.plot(
        execution_times_df['Group Size (i)'], 
        execution_times_df['Force Upper (FU)'], 
        marker='o', label='Force Upper (FU)', color='green'
    )
    plt.plot(
        execution_times_df['Group Size (i)'], 
        execution_times_df['Force Lower (FL)'], 
        marker='s', label='Force Lower (FL)', color='red'
    )
    plt.xlabel('Group Size (i)')
    plt.ylabel('Force Values')
    plt.title('Force Upper (FU) and Force Lower (FL) by Group Size')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return execution_times_df

execution_times_df = analyse_time_and_res()
