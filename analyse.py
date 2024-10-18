import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

path = './CICIoV2024/decimal/'
export_path_decimal = './CICIoV2024/tests/decimal/'

df_DoS = pd.read_csv('./CICIoV2024/decimal/decimal_DoS.csv')
df_Benign = pd.read_csv('./CICIoV2024/decimal/decimal_benign.csv')
df_spoofing_Gas = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-GAS.csv')
df_spoofing_RPM = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-RPM.csv')
df_spoofing_speed = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-SPEED.csv')
df_spoofing_steering = pd.read_csv('./CICIoV2024/decimal/decimal_spoofing-STEERING_WHEEL.csv')
df = pd.concat([df_DoS, df_Benign, df_spoofing_Gas, df_spoofing_RPM, df_spoofing_speed, df_spoofing_steering], ignore_index=True)
df.set_index('ID', inplace=True)
# print(df)

# Attack detection

df_attack = df[df['label'] == 'ATTACK']
del df_attack['category']
del df_attack['specific_class']
df_attack.reset_index(inplace=True)
del df_attack['ID']
print(df_attack)

df_benign = pd.read_csv(f'{path}decimal_benign.csv')
del df_benign['category']
del df_benign['specific_class']
del df_benign['ID']


# Export df_attack in csv
def export():
    df_attack.to_csv('./CICIoV2024/decimal/decimal_attack.csv', index=False)
    df_benign.to_csv('./CICIoV2024/decimal/decimal_benign_clean.csv', index=False)

def export_rename(name):
    df_attack.to_csv(f'./CICIoV2024/decimal/{name}.csv', index=False)
    df_benign.to_csv(f'./CICIoV2024/decimal/{name}.csv', index=False)


def export_sorted_atk():
    new_df = pd.DataFrame()
    for i in range(0, len(df_attack.columns)-1):
        new_df = pd.concat([new_df, df_attack[df_attack.columns[i]]], axis=1)
    new_df['label'] = 'ATTACK'
    new_df.drop_duplicates(inplace=True)

    new_df.to_csv('./CICIoV2024/decimal/decimal_attack_sorted.csv', index=False)

# For each column, addition the values from DATA_0 to DATA_7 and export the result in a new column named 'DATA_SUM'
def sum_data(dfs, name):
    dfs.drop_duplicates(inplace=True)
    cols_to_cast = dfs.columns[:-1]
    dfs['DATA_SUM'] = dfs[cols_to_cast].sum(axis=1)
    dfs.to_csv(f'{export_path_decimal}{name}_sum.csv', index=False)

    return dfs

# Concat of df_benign and df_attack
full_df_analysis = pd.concat([df_benign, df_attack], ignore_index=True)
# df_test.to_csv('./CICIoV2024/decimal/decimal_test.csv', index=False)

sum_data(df_attack, 'decimal_attack')
full_sum = sum_data(full_df_analysis, 'full_df')
# Sort full_df by DATA_SUM
full_sum.to_csv(f'{export_path_decimal}full_df_sum.csv', index=False)
full_sum.sort_values(by='DATA_SUM', inplace=True)
full_sum.to_csv(f'{export_path_decimal}full_df_sorted.csv', index=False)

# Graphical representation of the DATA_SUM : red = ATTACK, blue = BENIGN
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=full_sum, x=full_sum.index, y='DATA_SUM', hue='label', palette=['blue', 'red'])
plt.title('DATA_SUM')
plt.xlabel('Index')
plt.ylabel('DATA_SUM')
# plt.savefig(f'{export_path_decimal}full_df_sorted.png')
plt.show()

sns.set_theme(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=full_sum, x=full_sum.index, y='DATA_SUM', hue='label', palette=['blue', 'red'])
plt.title('DATA_SUM')
plt.xlabel('Index')
plt.ylabel('DATA_SUM')
# plt.savefig(f'{export_path_decimal}full_df_sorted.png')
plt.show()

# full_sum['label'] = full_sum['label'].replace({'ATTACK': 1, 'BENIGN': 0})
# full_sum.to_csv(f'{export_path_decimal}full_df_sum.csv', index=False)

# CAN-BUS Protocol
