import pandas as pd

path = "/home/dcunhrya/vista_bench/vista_bench/tb_v1_0_tb_classification_tasks/early_stage_management_answer.csv"
df = pd.read_csv(path)
print("Num columns:", len(df.columns))
print("Columns:")
for c in df.columns:
    print("-", c)