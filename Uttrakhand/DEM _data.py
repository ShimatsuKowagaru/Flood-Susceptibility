
import os
import numpy as np
import pandas as pd
import rasterio

from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from pysheds.grid import Grid


# ==================================================
# 1. FILE PATHS
# ==================================================

dem_file = r"D:\Flood Susceptibility\Uttrakhand\DEM_UK.tif"

rainfall_features_file = (
    r"D:\Flood Susceptibility\Uttrakhand\Rainfall_features.csv"
)

dem_features_file = (
    r"D:\Flood Susceptibility\Uttrakhand\DEM_features.csv"
)

combined_output_file = (
    r"D:\Flood Susceptibility\Combined_features.csv"
)

temp_folder = (
    r"D:\Flood Susceptibility\Uttrakhand\DEM_temp"
)

os.makedirs(temp_folder, exist_ok=True)

fine_dem_file = os.path.join(
    temp_folder, "DEM_1km.tif"
)

NODATA = -9999.0

# Intermediate DEM resolution in metres
FINE_RES = 1000

# Projected CRS for intermediate processing
FINE_CRS = "EPSG:32644"


# ==================================================
# 2. READ EXISTING RAINFALL FEATURES
# ==================================================

print("Reading rainfall feature dataset...")

rainfall_df = pd.read_csv(
    rainfall_features_file
)

required = {"longitude", "latitude"}

if not required.issubset(rainfall_df.columns):
    raise ValueError(
        "Rainfall feature CSV must contain longitude "
        "and latitude columns."
    )

rainfall_df["longitude"] = pd.to_numeric(
    rainfall_df["longitude"], errors="coerce"
)

rainfall_df["latitude"] = pd.to_numeric(
    rainfall_df["latitude"], errors="coerce"
)

if rainfall_df[
    ["longitude", "latitude"]
].isna().any().any():
    raise ValueError(
        "Rainfall feature data contains invalid coordinates."
    )

# Preserve the rainfall records and timestamps.
# Unique coordinates are used only to build the target grid.
rainfall_lons = np.sort(
    rainfall_df["longitude"].unique()
)

rainfall_lats = np.sort(
    rainfall_df["latitude"].unique()
)

if len(rainfall_lons) < 2 or len(rainfall_lats) < 2:
    raise ValueError(
        "At least two unique longitudes and latitudes "
        "are required."
    )

lon_diff = np.diff(rainfall_lons)
lat_diff = np.diff(rainfall_lats)

lon_res = float(np.median(lon_diff))
lat_res = float(np.median(lat_diff))

if (
    not np.allclose(lon_diff, lon_res, rtol=1e-3, atol=1e-6)
    or not np.allclose(lat_diff, lat_res, rtol=1e-3, atol=1e-6)
):
    raise ValueError(
        "The rainfall coordinates are not a regular grid. "
        "A regular-grid raster cannot safely be constructed "
        "from these coordinates without further processing."
    )

height = len(rainfall_lats)
width = len(rainfall_lons)

rainfall_transform = from_origin(
    rainfall_lons.min() - lon_res / 2,
    rainfall_lats.max() + lat_res / 2,
    lon_res,
    lat_res
)

print("Rainfall rows:", len(rainfall_df))
print("Unique spatial cells:", width * height)
print("Grid dimensions:", width, "x", height)
print("Longitude resolution:", lon_res)
print("Latitude resolution:", lat_res)


# ==================================================
# 3. REPROJECT DEM TO A FINER METRIC GRID
# ==================================================

print("\nPreparing intermediate DEM...")

with rasterio.open(dem_file) as src:

    if src.crs is None:
        raise ValueError("The DEM has no defined CRS.")

    bounds = transform_bounds(
        src.crs,
        FINE_CRS,
        *src.bounds,
        densify_pts=21
    )

    left, bottom, right, top = bounds

    fine_width = int(
        np.ceil((right - left) / FINE_RES)
    )

    fine_height = int(
        np.ceil((top - bottom) / FINE_RES)
    )

    fine_transform = from_origin(
        left, top, FINE_RES, FINE_RES
    )

    fine_dem = np.full(
        (fine_height, fine_width),
        NODATA,
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=fine_dem,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=fine_transform,
        dst_crs=FINE_CRS,
        dst_nodata=NODATA,
        resampling=Resampling.average
    )

valid = (
    np.isfinite(fine_dem)
    & (fine_dem != NODATA)
)

if not valid.any():
    raise ValueError(
        "No valid DEM cells after reprojection."
    )

with rasterio.open(
    fine_dem_file,
    "w",
    driver="GTiff",
    height=fine_height,
    width=fine_width,
    count=1,
    dtype="float32",
    crs=FINE_CRS,
    transform=fine_transform,
    nodata=NODATA,
    compress="lzw"
) as dst:

    dst.write(fine_dem, 1)

print("Intermediate DEM:", fine_dem.shape)
print("Valid DEM pixels:", int(valid.sum()))


# ==================================================
# 4. CALCULATE SLOPE
# ==================================================

print("\nCalculating slope...")

dem_gradient = fine_dem.copy()

mean_elevation = float(
    np.mean(dem_gradient[valid])
)

# Temporary filling is only for numerical gradients.
dem_gradient[~valid] = mean_elevation

dz_dy, dz_dx = np.gradient(
    dem_gradient,
    FINE_RES,
    FINE_RES
)

slope_radians = np.arctan(
    np.sqrt(dz_dx ** 2 + dz_dy ** 2)
)

slope_degrees = np.degrees(slope_radians)

slope_degrees[~valid] = np.nan

del dem_gradient, dz_dx, dz_dy


# ==================================================
# 5. HYDROLOGICAL PROCESSING
# ==================================================

print("\nInitializing Pysheds...")

grid = Grid.from_raster(fine_dem_file)

dem = grid.read_raster(fine_dem_file)

print("Filling pits...")

pit_filled = grid.fill_pits(dem)

print("Filling depressions...")

depression_filled = grid.fill_depressions(
    pit_filled
)

print("Resolving flats...")

inflated = grid.resolve_flats(
    depression_filled
)

print("Calculating flow direction...")

flowdir = grid.flowdir(inflated)

print("Calculating flow accumulation...")

flowacc = grid.accumulation(flowdir)


# Convert outputs to NumPy arrays while preserving
# masked cells as NaN.
def to_float_array(data):
    return np.asarray(
        np.ma.filled(
            np.ma.asarray(data, dtype=np.float64),
            np.nan
        ),
        dtype=np.float64
    )


flowdir_array = to_float_array(flowdir)
flowacc_array = to_float_array(flowacc)

# Do not treat missing or non-finite terrain cells
# as valid hydrological outputs.
flowdir_array[~valid] = np.nan
flowacc_array[~valid] = np.nan


# ==================================================
# 6. CALCULATE TWI
# ==================================================

print("Calculating TWI...")

tan_slope = np.tan(
    slope_radians.astype(np.float64)
)

valid_twi = (
    valid
    & np.isfinite(flowacc_array)
    & np.isfinite(tan_slope)
    & (tan_slope > 0)
)

# Approximate specific catchment area for a square
# projected grid. Flow accumulation is in cell counts.
specific_area = (
    flowacc_array + 1
) * FINE_RES

twi = np.full(
    fine_dem.shape,
    np.nan,
    dtype=np.float32
)

twi[valid_twi] = np.log(
    specific_area[valid_twi]
    / tan_slope[valid_twi]
).astype(np.float32)


# ==================================================
# 7. HELPER TO AGGREGATE RASTERS TO RAINFALL GRID
# ==================================================

def aggregate_to_rainfall_grid(
    source_array,
    method=Resampling.average,
    source_nodata=NODATA
):

    destination = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    source = np.where(
        np.isfinite(source_array),
        source_array,
        source_nodata
    ).astype(np.float32)

    reproject(
        source=source,
        destination=destination,
        src_transform=fine_transform,
        src_crs=FINE_CRS,
        src_nodata=source_nodata,
        dst_transform=rainfall_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=method
    )

    return destination


# ==================================================
# 8. MAP EACH DEM FEATURE TO RAINFALL RESOLUTION
# ==================================================

print("\nMapping DEM features to rainfall grid...")

elevation_target = aggregate_to_rainfall_grid(
    fine_dem,
    Resampling.average
)

slope_target = aggregate_to_rainfall_grid(
    slope_degrees,
    Resampling.average
)

# Flow direction is categorical. Nearest is used
# only to obtain a representative direction code.
flowdir_target = aggregate_to_rainfall_grid(
    flowdir_array,
    Resampling.nearest
)

flowacc_target = aggregate_to_rainfall_grid(
    flowacc_array,
    Resampling.average
)

twi_target = aggregate_to_rainfall_grid(
    twi,
    Resampling.average
)


# ==================================================
# 9. CREATE DEM FEATURE TABLE
# ==================================================

print("\nCreating DEM feature table...")

# Raster rows run north to south.
lon_grid, lat_grid = np.meshgrid(
    rainfall_lons,
    rainfall_lats[::-1]
)

dem_features_df = pd.DataFrame({
    "longitude": lon_grid.ravel(),
    "latitude": lat_grid.ravel(),
    "elevation": elevation_target.ravel(),
    "slope": slope_target.ravel(),
    "flow_direction": flowdir_target.ravel(),
    "flow_accumulation": flowacc_target.ravel(),
    "TWI": twi_target.ravel()
})

dem_features_df = dem_features_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Keep all grid cells; don't drop rows with missing values.
dem_features_df.to_csv(
    dem_features_file,
    index=False
)

print("DEM features saved:", dem_features_file)


# ==================================================
# 10. MAP FEATURES TO EXISTING RAINFALL RECORDS
# ==================================================

print("\nMerging terrain features with rainfall features...")

# Ensure each coordinate occurs only once in the DEM table.
if dem_features_df.duplicated(
    ["longitude", "latitude"]
).any():
    raise ValueError(
        "Duplicate coordinates found in DEM feature table."
    )

combined_df = rainfall_df.merge(
    dem_features_df,
    on=["longitude", "latitude"],
    how="left",
    validate="many_to_one",
    indicator=True
)

unmatched = (
    combined_df["_merge"] == "left_only"
).sum()

combined_df = combined_df.drop(
    columns=["_merge"]
)

# Confirm that no rainfall records were lost.
if len(combined_df) != len(rainfall_df):
    raise RuntimeError(
        "The number of rainfall records changed during merge."
    )

combined_df.to_csv(
    combined_output_file,
    index=False
)


# ==================================================
# 11. VALIDATION
# ==================================================

print("\n====================================")
print("PROCESSING COMPLETE")
print("====================================")

print("Rainfall records:", len(rainfall_df))
print("DEM grid cells:", len(dem_features_df))
print("Combined records:", len(combined_df))
print("Unmatched rainfall records:", unmatched)

print("\nMissing terrain values in combined data:")

terrain_columns = [
    "elevation",
    "slope",
    "flow_direction",
    "flow_accumulation",
    "TWI"
]

print(combined_df[terrain_columns].isna().sum())

print("\nCombined dataset preview:")
print(combined_df.head())

print("\nCombined dataset shape:")
print(combined_df.shape)

print("\nCombined file:")
print(combined_output_file)
