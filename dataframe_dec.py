import pandas as pd

# Decimal
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
df_combined = pd.concat([df_atk, df_benign], ignore_index=True)
df_combined = df_combined.drop(columns=['ID','label','specific_class', 'category'])



