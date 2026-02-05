import pandas as pd

# path = "/home/rdcunha/vista_project/vista_bench/radiation_treatment_outcomes_v1_1/radiation_outcome.csv"
path = "/home/rdcunha/vista_project/vista_bench/tb_v1_1_tb_classification_tasks/progression_assessment_discussed_subsampled_no_img_report.csv"
df = pd.read_csv(path)
print(len(df))
# print("Num columns:", len(df.columns))
# print("Columns:")
# for c in df.columns:
#     print("-", c)
print(df['report'].head().tolist())