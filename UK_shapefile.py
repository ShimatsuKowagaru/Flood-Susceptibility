import pandas as pd

input_file = r"D:\Flood Susceptibility\imerg_uttarakhand_2020_2025.csv"

df = pd.read_csv(input_file)

print(df.columns.tolist())
print(df.head())