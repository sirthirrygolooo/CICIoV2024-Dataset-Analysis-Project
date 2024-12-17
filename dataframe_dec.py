import pandas as pd
from functions import aggregate_columns

df_DoS = pd.read_csv('./CICIoV2024/decimal/decimal_DoS.csv')
df_spoofing_Gas = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-GAS.csv')
df_spoofing_RPM = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-RPM.csv')
df_spoofing_speed = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-SPEED.csv')
df_spoofing_steering = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-STEERING_WHEEL.csv')
df_benign = pd.read_csv('./CICIoV2024/decimal/decimal_benign.csv')


df_DoS['is_spoofing'] = 0
df_spoofing_Gas['is_spoofing'] = 1
df_spoofing_RPM['is_spoofing'] = 1
df_spoofing_speed['is_spoofing'] = 1
df_spoofing_steering['is_spoofing'] = 1
df_benign['is_spoofing'] = 0

df_atk = pd.concat(
    [df_DoS, df_spoofing_Gas, df_spoofing_RPM, df_spoofing_speed, df_spoofing_steering], 
    ignore_index=True
)

df_atk['isMechant'] = 1
df_benign['isMechant'] = 0

df_atk_clean = df_atk.drop(columns=['label', 'specific_class', 'category'])
df_benign_clean = df_benign.drop(columns=['label', 'specific_class', 'category'])

df_combined = pd.concat([df_atk, df_benign], ignore_index=True)
df_combined = df_combined.drop(columns=['label', 'specific_class', 'category'])

aggregated_df_atk = aggregate_columns(df_atk_clean, id_column='ID')
aggregated_df_benign = aggregate_columns(df_benign_clean, id_column='ID')

aggregated_df = pd.concat([aggregated_df_atk, aggregated_df_benign], ignore_index=True)
aggregated_df = aggregated_df.drop(columns=['ID'])

def prepare_full_df():
    df_atk['spoofing_type'] = 0
    df_atk.loc[df_atk['specific_class'] == 'GAS', 'type_spoofing'] = 1
    df_atk.loc[df_atk['specific_class'] == 'RPM', 'type_spoofing'] = 2
    df_atk.loc[df_atk['specific_class'] == 'SPEED', 'type_spoofing'] = 3
    df_atk.loc[df_atk['specific_class'] == 'STEERING_WHEEL', 'type_spoofing'] = 4

    df_benign['spoofing_type'] = 0

    df_atk.drop(columns=['label', 'specific_class', 'category'], inplace=True)
    df_benign.drop(columns=['label', 'specific_class', 'category'], inplace=True)

    df_atk.to_csv('./CICIoV2024/tests/decimal/decimal_full_atk.csv', index=False)
    df_benign.to_csv('./CICIoV2024/tests/decimal/decimal_full_benign.csv', index=False)


try :
    full_df_atk = pd.read_csv('./CICIoV2024/tests/decimal/decimal_full_atk.csv')
    full_df_benign = pd.read_csv('./CICIoV2024/tests/decimal/decimal_full_benign.csv')
except FileNotFoundError:
    print('[!] full_df files not found, creating it... It can take few minutes')
    prepare_full_df()
    full_df_atk = pd.read_csv('./CICIoV2024/tests/decimal/decimal_full_atk.csv')
    full_df_benign = pd.read_csv('./CICIoV2024/tests/decimal/decimal_full_benign.csv')


    
