
import pandas as pd
import numpy as np

# File paths
daily_path = r"D:\Flood Susceptibility\combined_features_daily.csv"
events_path = r"D:\Flood Susceptibility\events_places_geocoded.csv"

output_path = r"D:\Flood Susceptibility\Combined_features_daily_labeled.csv"
registry_path = r"D:\Flood Susceptibility\event_location_registry.csv"
matched_path = r"D:\Flood Susceptibility\matched_flood_events.csv"
unmatched_path = r"D:\Flood Susceptibility\unmatched_flood_events.csv"

# Settings
MAX_DISTANCE_KM = 8.0
MIN_CONFIDENCE = {"A", "B"}

# Read files
daily = pd.read_csv(daily_path)
events = pd.read_csv(events_path)

daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
events["date_start"] = pd.to_datetime(
    events["date_start"], errors="coerce"
).dt.normalize()
events["date_end"] = pd.to_datetime(
    events["date_end"], errors="coerce"
).dt.normalize()

# Convert coordinates to numeric
for col in ["latitude", "longitude"]:
    daily[col] = pd.to_numeric(daily[col], errors="coerce")

for col in ["lat", "lon"]:
    events[col] = pd.to_numeric(events[col], errors="coerce")

# Keep every source location in the registry
registry = events.copy()
registry["matched_grid_latitude"] = np.nan
registry["matched_grid_longitude"] = np.nan
registry["match_distance_km"] = np.nan
registry["match_result"] = "pending"
registry["match_reason"] = ""

# Unique coordinate pairs from the daily grid
grid = (
    daily[["latitude", "longitude"]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)

grid_lat = grid["latitude"].to_numpy()
grid_lon = grid["longitude"].to_numpy()

# Haversine distance between one event and all grid points
def distances_km(lat, lon):
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(grid_lat)
    lon2 = np.radians(grid_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371.0088 * 2 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

# Prepare accepted event matches
matches = []
matched_event_indices = set()

for i, event in events.iterrows():

    # Preserve excluded records, but don't use them as labels
    if str(event["status"]).strip().lower() != "include":
        registry.loc[i, "match_result"] = "not_used"
        registry.loc[i, "match_reason"] = "Event status is not include"
        continue

    if pd.isna(event["lat"]) or pd.isna(event["lon"]):
        registry.loc[i, "match_result"] = "unmatched"
        registry.loc[i, "match_reason"] = "Missing event coordinates"
        continue

    confidence = str(event.get("geocode_confidence", "")).strip().upper()

    if confidence not in MIN_CONFIDENCE:
        registry.loc[i, "match_result"] = "review"
        registry.loc[i, "match_reason"] = "Low or missing geocode confidence"
        continue

    district_match = str(event.get("district_match", "")).strip().lower()

    if district_match not in {"true", "1", "1.0"}:
        registry.loc[i, "match_result"] = "review"
        registry.loc[i, "match_reason"] = "District match not confirmed"
        continue

    if pd.isna(event["date_start"]) or pd.isna(event["date_end"]):
        registry.loc[i, "match_result"] = "unmatched"
        registry.loc[i, "match_reason"] = "Missing event dates"
        continue

    if event["date_end"] < event["date_start"]:
        registry.loc[i, "match_result"] = "unmatched"
        registry.loc[i, "match_reason"] = "Invalid event date range"
        continue

    # Find the nearest rainfall grid coordinate
    distances = distances_km(event["lat"], event["lon"])
    nearest_idx = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_idx])

    registry.loc[i, "match_distance_km"] = nearest_distance

    if nearest_distance > MAX_DISTANCE_KM:
        registry.loc[i, "match_result"] = "unmatched"
        registry.loc[i, "match_reason"] = "Nearest grid outside distance limit"
        continue

    grid_latitude = float(grid_lat[nearest_idx])
    grid_longitude = float(grid_lon[nearest_idx])

    registry.loc[i, "matched_grid_latitude"] = grid_latitude
    registry.loc[i, "matched_grid_longitude"] = grid_longitude

    # Generate an event match for every day in its date range
    dates = pd.date_range(
        event["date_start"], event["date_end"], freq="D"
    )

    for date in dates:
        matches.append({
            "date": date,
            "latitude": round(grid_latitude, 6),
            "longitude": round(grid_longitude, 6),
            "event_id": str(event["event_id"]),
            "place_id": str(event["place_id"]),
            "place": str(event["place"]),
            "event_lat": float(event["lat"]),
            "event_lon": float(event["lon"]),
            "match_distance_km": nearest_distance
        })

    registry.loc[i, "match_result"] = "matched_to_grid"
    registry.loc[i, "match_reason"] = "Passed filters and distance threshold"
    matched_event_indices.add(i)

# Build the matched-event table
match_columns = [
    "date", "latitude", "longitude", "event_id", "place_id",
    "place", "event_lat", "event_lon", "match_distance_km"
]

match_df = pd.DataFrame(matches, columns=match_columns)

# Merge multiple events at the same grid coordinate and date
if not match_df.empty:
    match_df["date"] = pd.to_datetime(match_df["date"]).dt.normalize()

    match_df["latitude"] = match_df["latitude"].round(6)
    match_df["longitude"] = match_df["longitude"].round(6)

    match_df["event_coordinates"] = (
        match_df["place"] + " (" +
        match_df["event_lat"].round(6).astype(str) + ", " +
        match_df["event_lon"].round(6).astype(str) + ")"
    )

    grouped = match_df.groupby(
        ["date", "latitude", "longitude"], as_index=False
    ).agg(
        flood_event_ids=("event_id", lambda x: "; ".join(sorted(set(x)))),
        flood_places=("place", lambda x: "; ".join(sorted(set(x)))),
        flood_event_coordinates=(
            "event_coordinates", lambda x: "; ".join(sorted(set(x)))
        ),
        matched_event_count=("event_id", "nunique"),
        nearest_event_distance_km=("match_distance_km", "min")
    )
else:
    grouped = pd.DataFrame(columns=[
        "date", "latitude", "longitude", "flood_event_ids",
        "flood_places", "flood_event_coordinates",
        "matched_event_count", "nearest_event_distance_km"
    ])

# Create matching keys
daily["latitude"] = daily["latitude"].round(6)
daily["longitude"] = daily["longitude"].round(6)

# No event match does NOT automatically mean confirmed non-flood
daily["flood_label"] = -1

# Attach event details to daily features
daily = daily.merge(
    grouped,
    on=["date", "latitude", "longitude"],
    how="left",
    validate="many_to_one"
)

# Positive labels only where a dated, spatial event match exists
daily.loc[
    daily["matched_event_count"].notna(), "flood_label"
] = 1

daily["matched_event_count"] = (
    daily["matched_event_count"].fillna(0).astype(int)
)

# Export all results
daily.to_csv(output_path, index=False)
registry.to_csv(registry_path, index=False)
match_df.to_csv(matched_path, index=False)
registry[
    registry["match_result"].isin(["unmatched", "review"])
].to_csv(unmatched_path, index=False)

# Summary
print("Daily dataset rows:", len(daily))
print("Positive labeled rows:", (daily["flood_label"] == 1).sum())
print("Unknown/unlabeled rows:", (daily["flood_label"] == -1).sum())
print("Unique matched events:", len(matched_event_indices))
print("Location registry rows:", len(registry))
print("Review/unmatched locations:", len(
    registry[registry["match_result"].isin(["unmatched", "review"])]
))

print("\nSaved files:")
print(output_path)
print(registry_path)
print(matched_path)
print(unmatched_path)
