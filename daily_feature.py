
import pandas as pd

# Read combined features
df = pd.read_csv(
    r"D:\Flood Susceptibility\Combined_features.csv"
)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Keep only observations recorded at exactly 00:00
daily = df[
    (df["date"].dt.hour == 0) &
    (df["date"].dt.minute == 0) &
    (df["date"].dt.second == 0)
].copy()

# Sort by date and grid coordinates
daily = daily.sort_values(
    ["date", "latitude", "longitude"]
)

# Save the result
daily.to_csv(
    r"D:\Flood Susceptibility\Combined_features_daily.csv",
    index=False
)

# Display results
print("Original rows:", len(df))
print("Rows after filtering:", len(daily))
print(daily.head(10))
