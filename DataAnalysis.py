import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

df_DoS = pd.read_csv('./CICIoV2024/decimal/decimal_DoS.csv')
df_Benign = pd.read_csv('./CICIoV2024/decimal/decimal_benign.csv')
df_spoofing_Gas = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-GAS.csv')
df_spoofing_RPM = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-RPM.csv')
df_spoofing_speed = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-SPEED.csv')
df_spoofing_steering = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-STEERING_WHEEL.csv')
df = pd.concat([df_DoS, df_Benign, df_spoofing_Gas, df_spoofing_RPM, df_spoofing_speed, df_spoofing_steering], ignore_index=True)
df.set_index('ID', inplace=True)

df_attack = df[df['label'] == 'ATTACK']

print(df_attack)
atk_view = df_attack.describe()
print(atk_view)

benign_view = df_Benign.describe()
print(benign_view)

plt.figure(figsize=(10, 6))
sns.countplot(data=df_attack, x='DATA_0')
plt.title("Count Plot of Values in attack_df DATA_0")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df_Benign, x='DATA_0')
plt.title("Count Plot of Values in benign_df DATA_0")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Full dump of df_Benign ordered by id

print(df_Benign.sort_values(by='ID'))
print(df_Benign)
print(df_Benign['DATA_0'].value_counts())
print(df_Benign['DATA_1'].value_counts())
print(df_Benign['DATA_2'].value_counts())
print(df_Benign['DATA_3'].value_counts())
print(df_Benign['DATA_4'].value_counts())
print(df_Benign['DATA_5'].value_counts())
print(df_Benign['DATA_6'].value_counts())
print(df_Benign['DATA_7'].value_counts())
