"""
Geocode events_places.csv with OpenStreetMap Nominatim.

Setup:
    pip install geopy pandas

Run:
    python geocode_places.py

Inputs : events_places.csv            (from parse_flood_events.py)
         manual_coords.csv  (optional) columns: place_id,lat,lon  -> overrides everything
Outputs: events_places_geocoded.csv   all rows, with lat/lon where found
         events_places_check.csv      rows that failed or look doubtful -> fix by hand
         geocode_cache.json           cache so reruns don't repeat requests

Nominatim policy: max 1 request/second, use a real user_agent. ~300 rows takes
roughly 5-10 minutes the first time.
"""
import json
import time
from pathlib import Path

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.point import Point

USER_AGENT = "aman2006bansal@gmail.com"  # change to include your email/name
IN_CSV = "events_places.csv"
MANUAL_CSV = "manual_coords.csv"
CACHE_FILE = Path("geocode_cache.json")

# Uttarakhand bounding box (generous): lat 28.7-31.5, lon 77.5-81.1
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 28.7, 31.5, 77.5, 81.1
VIEWBOX = [Point(LAT_MAX, LON_MIN), Point(LAT_MIN, LON_MAX)]

SMALL_PLACE = {"village", "hamlet", "locality", "isolated_dwelling", "neighbourhood",
               "suburb", "stream", "river", "canal", "drain", "peak", "valley", "farm"}
TOWN = {"town", "city", "municipality", "census_town"}


def in_bbox(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def grade(raw):
    """A = village/stream level, B = town, C = admin area only."""
    cls, typ = raw.get("class", ""), raw.get("type", "")
    if cls == "waterway" or typ in SMALL_PLACE:
        return "A"
    if typ in TOWN:
        return "B"
    return "C"


def main():
    df = pd.read_csv(IN_CSV)
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=15)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1,
                          max_retries=2, error_wait_seconds=5)
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

    def lookup(query):
        if query in cache:
            return cache[query]
        try:
            res = geocode(query, exactly_one=False, limit=5, country_codes="in",
                          viewbox=VIEWBOX, bounded=True, addressdetails=False)
        except Exception as e:  # network or service error: don't cache
            print(f"  error for {query!r}: {e}")
            return []
        out = [{"lat": r.latitude, "lon": r.longitude, "name": r.address, "raw": r.raw}
               for r in (res or [])]
        cache[query] = out
        CACHE_FILE.write_text(json.dumps(cache))
        return out

    rows = []
    todo = df[df.status != "exclude"]
    total = len(todo)
    for n, (_, r) in enumerate(todo.iterrows(), 1):
        place, district = r["place"], r["district"]
        dkey = str(district).split("/")[0].split()[0].lower()  # "Pauri Garhwal" -> "pauri"
        attempts = [f"{place}, {district}, Uttarakhand, India",
                    f"{place}, Uttarakhand, India"]
        best = None
        for q in attempts:
            cands = [c for c in lookup(q) if in_bbox(c["lat"], c["lon"])]
            if not cands:
                continue
            # prefer a candidate whose address mentions the right district
            cands.sort(key=lambda c: dkey not in c["name"].lower())
            best = (q, cands[0])
            break
        if best:
            q, c = best
            district_match = dkey in c["name"].lower()
            conf = grade(c["raw"])
            rows.append(dict(place_id=r["place_id"], lat=c["lat"], lon=c["lon"],
                             geocode_confidence=conf, matched_name=c["name"],
                             osm_class=c["raw"].get("class"), osm_type=c["raw"].get("type"),
                             district_match=district_match, query_used=q))
        else:
            rows.append(dict(place_id=r["place_id"], lat=None, lon=None,
                             geocode_confidence="", matched_name="", osm_class="",
                             osm_type="", district_match=False, query_used=""))
        if n % 25 == 0:
            print(f"{n}/{total} done")

    g = pd.DataFrame(rows)
    out = df.drop(columns=["lat", "lon", "geocode_confidence"], errors="ignore").merge(
        g, on="place_id", how="left")

    # manual overrides
    if Path(MANUAL_CSV).exists():
        m = pd.read_csv(MANUAL_CSV).set_index("place_id")
        for pid, row in m.iterrows():
            idx = out.index[out.place_id == pid]
            out.loc[idx, ["lat", "lon"]] = row["lat"], row["lon"]
            out.loc[idx, "geocode_confidence"] = "M"
            out.loc[idx, "district_match"] = True
        print(f"applied {len(m)} manual overrides")

    out.to_csv("events_places_geocoded.csv", index=False)

    use = out[out.status != "exclude"]
    check = use[use.lat.isna() | (use.geocode_confidence == "C") | (~use.district_match.fillna(False))]
    check.to_csv("events_places_check.csv", index=False)

    print("\n--- summary (non-excluded rows) ---")
    print(f"total place rows      : {len(use)}")
    print(f"found                 : {use.lat.notna().sum()}")
    print(f"not found             : {use.lat.isna().sum()}")
    print("confidence grades     :", use.geocode_confidence.value_counts().to_dict())
    print(f"district mismatch     : {(use.lat.notna() & ~use.district_match.fillna(False)).sum()}")
    print(f"rows to check by hand : {len(check)}  -> events_places_check.csv")


if __name__ == "__main__":
    main()
