import pandas as pd


df = pd.read_csv(r"D:\Flood Susceptibility\Combined_features.csv", nrows=100)

print(df.to_string())