import pandas as pd

label_df = pd.read_csv("E:/LUNA16/csvfiles/annotations.csv")

small_df = label_df[label_df["diameter_mm"]<=16]

print(small_df.shape)