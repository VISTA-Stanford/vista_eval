import pandas as pd

path = "/home/dcunhrya/vista_bench/radiation_treatment_outcomes_v1_1/radiation_outcome.csv"
df = pd.read_csv(path)
print(len(df))
print("Num columns:", len(df.columns))
print("Columns:")
for c in df.columns:
    print("-", c)
# print(df['local_path'].tolist())