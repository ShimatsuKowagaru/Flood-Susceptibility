import pandas as pd

rainfall_file = r"D:\Flood Susceptibility\Uttrakhand\Rainfall_features.csv"
dem_file = r"D:\Flood Susceptibility\Uttrakhand\DEM_features.csv"

# Read both CSV files
rainfall_df = pd.read_csv(rainfall_file)
dem_df = pd.read_csv(dem_file)

print("Rainfall data:")
print(rainfall_df.head())

print("\nDEM data:")
print(dem_df.head())

# Merge using latitude and longitude
df = pd.merge(
    rainfall_df,
    dem_df,
    on=["latitude", "longitude"],
    how="left"
)

# Save the combined dataset
output_file = r"D:\Flood Susceptibility\Uttrakhand\Combined_features.csv"

df.to_csv(output_file, index=False)

print("\nCombined data:")
print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())