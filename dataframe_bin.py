import pandas as pd
from functions import aggregate_binary_dataframe

df_DoS = pd.read_csv('./CICIoV2024/binary/binary_DoS.csv')
df_spoofing_Gas = pd.read_csv('./CICIoV2024/binary/binary_spoofing-GAS.csv')
df_spoofing_RPM = pd.read_csv('./CICIoV2024/binary/binary_spoofing-RPM.csv')
df_spoofing_speed = pd.read_csv('./CICIoV2024/binary/binary_spoofing-SPEED.csv')
df_spoofing_steering = pd.read_csv('./CICIoV2024/binary/binary_spoofing-STEERING_WHEEL.csv')

df_benign = pd.read_csv('./CICIoV2024/binary/binary_benign.csv')
df_atk = pd.concat([df_DoS, df_spoofing_Gas, df_spoofing_RPM, df_spoofing_speed, df_spoofing_steering], ignore_index=True)  

# Feature selection

df_atk['isMechant'] = 1
df_benign['isMechant'] = 0

df_atk_clean = df_atk.drop(columns=['label','specific_class', 'category'])
df_benign_clean = df_benign.drop(columns=['label','specific_class', 'category'])

df_combined = pd.concat([df_atk, df_benign], ignore_index=True)
# df_combined = df_combined.drop(columns=df_combined.filter(like='ID').columns)
df_combined = df_combined.drop(columns=['label','specific_class', 'category'])

df_combined.to_csv('./CICIoV2024/binary/good.csv', index=False)



aggregated_df_atk = aggregate_binary_dataframe(df_atk_clean)
aggregated_df_benign = aggregate_binary_dataframe(df_benign_clean)

aggregated_df = pd.concat([aggregated_df_atk,aggregated_df_benign], ignore_index=True)

aggregated_df = aggregated_df.drop(columns=aggregated_df.filter(like='ID').columns)