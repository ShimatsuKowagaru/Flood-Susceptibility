import pandas as pd
import numpy as np


# --------------------------------
# 1. FILE PATHS
# --------------------------------

input_file = r"D:\Flood Susceptibility\imerg_uttarakhand_2020_2025.csv"

output_file = r"D:\Flood Susceptibility\Uttrakhand\Rainfall_features.csv"


# --------------------------------
# 2. READ DATA
# --------------------------------

df = pd.read_csv(input_file)

print("Columns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())


# --------------------------------
# 3. CHANGE THESE IF NEEDED
# --------------------------------

lon_column = "longitude"
lat_column = "latitude"
time_column = "date"
rainfall_column = "precipitation"  # Change this if your CSV uses a different name for the rainfall column


# --------------------------------
# 4. CONVERT TIME
# --------------------------------

df[time_column] = pd.to_datetime(
    df[time_column]
)


# --------------------------------
# 5. SORT DATA
# --------------------------------

df = df.sort_values(
    by=[
        lon_column,
        lat_column,
        time_column
    ]
)


# --------------------------------
# 6. SET TIME AS INDEX
# --------------------------------

df = df.set_index(
    time_column
)


# --------------------------------
# 7. CALCULATE ACCUMULATED
#    RAINFALL
# --------------------------------

print("\nCalculating rainfall features...")


# 30-minute rainfall
df["rainfall_30min"] = (
    df[rainfall_column]
)


# 1-hour rainfall
df["rainfall_1h"] = (
    df.groupby(
        [lon_column, lat_column]
    )[rainfall_column]
    .rolling("1h")
    .sum()
    .reset_index(
        level=[0, 1],
        drop=True
    )
)


# 3-hour rainfall
df["rainfall_3h"] = (
    df.groupby(
        [lon_column, lat_column]
    )[rainfall_column]
    .rolling("3h")
    .sum()
    .reset_index(
        level=[0, 1],
        drop=True
    )
)


# 6-hour rainfall
df["rainfall_6h"] = (
    df.groupby(
        [lon_column, lat_column]
    )[rainfall_column]
    .rolling("6h")
    .sum()
    .reset_index(
        level=[0, 1],
        drop=True
    )
)


# 12-hour rainfall
df["rainfall_12h"] = (
    df.groupby(
        [lon_column, lat_column]
    )[rainfall_column]
    .rolling("12h")
    .sum()
    .reset_index(
        level=[0, 1],
        drop=True
    )
)


# 24-hour rainfall
df["rainfall_24h"] = (
    df.groupby(
        [lon_column, lat_column]
    )[rainfall_column]
    .rolling("24h")
    .sum()
    .reset_index(
        level=[0, 1],
        drop=True
    )
)


# --------------------------------
# 8. RESET INDEX
# --------------------------------

df = df.reset_index()


# --------------------------------
# 9. REMOVE ORIGINAL RAINFALL
#    COLUMN IF DESIRED
# --------------------------------

# Keep the original rainfall column
# because it represents the 30-minute
# rainfall.


# --------------------------------
# 10. SAVE
# --------------------------------

df.to_csv(
    output_file,
    index=False
)


# --------------------------------
# 11. DISPLAY
# --------------------------------

print("\n--------------------------------")
print("RAIN FALL PROCESSING COMPLETE")
print("--------------------------------")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head(20))

print("\nSaved to:")
print(output_file)