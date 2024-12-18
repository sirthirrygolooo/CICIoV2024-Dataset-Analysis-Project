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
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(
        min_split_gain=0.0,     # Gain minimum pour un split
        max_depth=10,           # Profondeur maximale de l'arbre
        min_child_samples=5,    # Nombre minimum d'échantillons pour un split
        learning_rate=0.1,      # Taux d'apprentissage
        n_estimators=100,
        force_col_wise=True,
        verbose=-1 ),
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
    i = 10
    execution_times = []

    models = {
        "XGBoost": XGBClassifier(eval_metric='logloss'),
    }

    FU_values = []
    FL_values = []

    while i < 300:
        aggregated_df_atk = aggregate_columns2(decimal.df_atk_clean, id_column='ID', group_size=i)
        aggregated_df_benign = aggregate_columns2(decimal.df_benign_clean, id_column='ID', group_size=i)

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

            # les forceux
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

    # Graphiqeuuu
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 1, 1)
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

    plt.subplot(3, 1, 2)
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

    plt.subplot(3, 1, 3)
    plt.grid(visible=True)
    plt.plot(
        execution_times_df['Group Size (i)'], 
        execution_times_df['Execution Time (s)'], 
        marker='^', label='Execution Time', color='purple'
    )
    plt.xlabel('Group Size (i)')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time by Group Size')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(execution_times_df[execution_times_df['No Misclassification'] == False])

    return execution_times_df

# aggregated_df_atk = aggregate_columns2(df=decimal.df_atk_clean, id_column='ID')
# aggregated_df_benign = aggregate_columns2(df=decimal.df_benign_clean, id_column='ID')

# aggregated_df = pd.concat([aggregated_df_atk, aggregated_df_benign], ignore_index=True)
# aggregated_df = aggregated_df.drop(columns=['ID'])

# aggregated_df.to_csv(f'{EXPORT_PATH_TESTS_DEC}hehe.csv', index=False)

# df = pd.read_csv(f'{EXPORT_PATH_TESTS_DEC}hehe.csv')



# diagnostic = diagnosticv1(models=models,df=df)

# diagnosticv2 = diagnosticv2(models=models,df=df)

try :
    full_df_aggregated = pd.read_csv(f'{EXPORT_PATH_TESTS_DEC}full_df_aggregated.csv')
except FileNotFoundError:
    print("[!] full_df_aggeated.csv not found, creating it... It can take few minutes")
    full_df_atk_aggregated = aggregate_columns2(decimal.full_df_atk, id_column='ID')
    full_df_benign_aggregated = aggregate_columns2(decimal.full_df_benign, id_column='ID')

    full_df_aggregated = pd.concat([full_df_atk_aggregated, full_df_benign_aggregated], ignore_index=True)

    full_df_aggregated.to_csv(f'{EXPORT_PATH_TESTS_DEC}full_df_aggregated.csv', index=False)

full_df = pd.concat([decimal.full_df_atk, decimal.full_df_benign], ignore_index=True)
# full_df.to_csv(f'{EXPORT_PATH_TESTS_DEC}full_df.csv', index=False)

diagnosticv1 = diagnosticv1(models=models,df=full_df)
diagnosticv2 = diagnosticv2(models=models,df=full_df)
# diagnosticv3 = diagnosticv3(models=models,df=full_df)
diganostic_final = diagnostic_final(models=models,df=full_df)