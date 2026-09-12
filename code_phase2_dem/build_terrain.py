"""Download USGS 3DEP 1-arc-second DEM and build county terrain features.

The source rasters are the USGS TNM ``National Elevation Dataset (NED) 1
arc-second`` products (about 30 m).  They are downloaded only for the
competition counties in Indiana, Ohio, Pennsylvania, and West Virginia.

The output is ``data/geo/county_terrain.csv`` with one row per competition
county.  ``terrain_ruggedness`` is the mean 3x3-neighbour mean absolute
elevation difference (meters), computed only where the neighbour is inside
the county and valid in the DEM.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
DBF = GEO_DIR / "c_16ap26.dbf"
SHP = GEO_DIR / "c_16ap26.shp"
DEM_DIR = GEO_DIR / "dem_3dep_1arcsec"
OUTPUT = GEO_DIR / "county_terrain.csv"
MANIFEST = DEM_DIR / "manifest.json"
TARGET_STATE_FIPS = {"18", "39", "42", "54"}
TNM_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"

sys.path.insert(0, str(ROOT / "code_phase2_dem"))
from spatial import read_dbf_fips_in_order, read_shp_polygons  # noqa: E402


def target_geometries():
    fips = read_dbf_fips_in_order(DBF)
    geometries = read_shp_polygons(SHP)
    out = {}
    for code, geom in zip(fips, geometries):
        if code[:2] in TARGET_STATE_FIPS and geom is not None:
            out[code] = geom
    return out


def query_tiles(geometries):
    latest = {}
    query_urls = []
    # TNM can time out on one large four-state bbox.  State-level requests
    # are equivalent for the target counties and are much more reliable.
    for state in sorted(TARGET_STATE_FIPS):
        state_geometries = {f: g for f, g in geometries.items() if f[:2] == state}
        bounds = np.asarray([g.bounds for g in state_geometries.values()], dtype=float)
        minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
        maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()
        params = {
            "datasets": "National Elevation Dataset (NED) 1 arc-second",
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "max": 10000,
        }
        url = TNM_URL + "?" + urllib.parse.urlencode(params)
        query_urls.append(url)
        payload = None
        error = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=240) as response:
                    payload = json.load(response)
                break
            except Exception as exc:  # pragma: no cover - remote retry path
                error = repr(exc)
                time.sleep(4 * attempt)
        if payload is None:
            raise RuntimeError(f"USGS query failed for state {state}: {error}")
        print(f"USGS state {state}: {payload.get('total', 0)} products", flush=True)
    # The service exposes historical revisions.  Keep the newest revision
    # for each 1x1-degree tile, keyed by its directory name (e.g. n40w080).
        for item in payload.get("items", []):
            download_url = item.get("downloadURL") or ""
            parts = download_url.rstrip("/").split("/")
            if len(parts) < 2 or not re.fullmatch(r"[ns]\d+[ew]\d+", parts[-2], re.I):
                continue
            tile = parts[-2].lower()
            date_match = re.search(r"(\d{8})\.tif$", download_url, re.I)
            date_key = int(date_match.group(1)) if date_match else 0
            rank = (date_key, str(item.get("lastUpdated", "")), download_url)
            if tile not in latest or rank > latest[tile][0]:
                latest[tile] = (rank, item)
    if not latest:
        raise RuntimeError(f"USGS returned no DEM tiles. Query URLs: {query_urls}")
    selected = []
    for tile, (_, item) in sorted(latest.items()):
        selected.append({
            "tile": tile,
            "url": item["downloadURL"],
            "sizeInBytes": int(item.get("sizeInBytes") or 0),
            "publicationDate": item.get("publicationDate"),
            "lastUpdated": item.get("lastUpdated"),
        })
    return selected, ";".join(query_urls)


def download_one(item):
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    destination = DEM_DIR / Path(urllib.parse.urlparse(item["url"]).path).name
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination, "existing"
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "INFORMS-2026-terrain/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024 * 8)
                    if not chunk:
                        break
                    out.write(chunk)
            if partial.stat().st_size < 1_000_000:
                raise RuntimeError(f"download unexpectedly small: {partial.stat().st_size} bytes")
            partial.replace(destination)
            return destination, f"downloaded_attempt_{attempt}"
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = repr(exc)
            if partial.exists():
                partial.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {item['url']}: {last_error}")


def download_tiles(tiles, workers):
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(download_one, item): item for item in tiles}
        for idx, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            path, status = future.result()
            completed.append({**item, "local_path": str(path), "status": status})
            print(f"[{idx}/{len(tiles)}] {item['tile']} {status}", flush=True)
    completed.sort(key=lambda x: x["tile"])
    return completed


def _new_accumulator():
    return {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf}


def _update(acc, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    acc["count"] += int(values.size)
    acc["sum"] += float(values.sum(dtype=np.float64))
    acc["sumsq"] += float(np.square(values).sum(dtype=np.float64))
    acc["min"] = min(acc["min"], float(values.min()))
    acc["max"] = max(acc["max"], float(values.max()))


def _finish(acc):
    if acc["count"] == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    mean = acc["sum"] / acc["count"]
    variance = max(0.0, acc["sumsq"] / acc["count"] - mean * mean)
    return {"mean": mean, "std": np.sqrt(variance), "min": acc["min"], "max": acc["max"]}


def county_stats(geometries, downloaded):
    stats = {
        fips: {key: _new_accumulator() for key in ("elev", "slope", "rugged")}
        for fips in geometries
    }
    for tile_idx, item in enumerate(downloaded, start=1):
        path = Path(item["local_path"])
        with rasterio.open(path) as src:
            raster_bounds = src.bounds
            src_crs = src.crs
            for fips, geom in geometries.items():
                minx, miny, maxx, maxy = geom.bounds
                if maxx < raster_bounds.left or minx > raster_bounds.right or maxy < raster_bounds.bottom or miny > raster_bounds.top:
                    continue
                shape = mapping(geom)
                if src_crs and str(src_crs).upper() not in {"EPSG:4269", "EPSG:4326"}:
                    shape = transform_geom("EPSG:4269", src_crs, shape)
                try:
                    data, transform = raster_mask(src, [shape], crop=True, pad=True, pad_width=1, filled=False)
                except ValueError:
                    continue
                elevation = data[0]
                valid = ~np.ma.getmaskarray(elevation)
                if not valid.any():
                    continue
                z = np.asarray(elevation.data, dtype=np.float64)
                fill_value = float(np.nanmedian(z[valid]))
                z[~valid] = fill_value
                values = z[valid]
                _update(stats[fips]["elev"], values)

                center_lat = float(transform.f + transform.e * (z.shape[0] / 2.0))
                dx_m = max(abs(transform.a) * 111320.0 * np.cos(np.deg2rad(center_lat)), 1.0)
                dy_m = max(abs(transform.e) * 110540.0, 1.0)
                dz_dy, dz_dx = np.gradient(z, dy_m, dx_m)
                slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
                _update(stats[fips]["slope"], slope[valid])

                # Mean absolute elevation difference to valid 8-neighbours.
                rough_sum = np.zeros_like(z, dtype=np.float64)
                rough_count = np.zeros_like(z, dtype=np.float64)
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        neighbour_z = np.roll(np.roll(z, di, axis=0), dj, axis=1)
                        neighbour_valid = np.roll(np.roll(valid, di, axis=0), dj, axis=1)
                        rough_sum += np.where(neighbour_valid, np.abs(z - neighbour_z), 0.0)
                        rough_count += neighbour_valid
                rugged = rough_sum / np.maximum(rough_count, 1.0)
                _update(stats[fips]["rugged"], rugged[valid & (rough_count > 0)])
        print(f"[terrain] processed tile {tile_idx}/{len(downloaded)}: {item['tile']}", flush=True)

    rows = []
    for fips in sorted(geometries):
        elev = _finish(stats[fips]["elev"])
        slope = _finish(stats[fips]["slope"])
        rugged = _finish(stats[fips]["rugged"])
        rows.append({
            "fipsCode": fips,
            "elevation_mean_m": elev["mean"],
            "elevation_std_m": elev["std"],
            "elevation_min_m": elev["min"],
            "elevation_max_m": elev["max"],
            "slope_mean_deg": slope["mean"],
            "slope_std_deg": slope["std"],
            "terrain_ruggedness": rugged["mean"],
        })
    result = pd.DataFrame(rows)
    if result.isna().any().any():
        raise RuntimeError("terrain summary contains missing values: " + str(result.isna().sum().to_dict()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    geometries = target_geometries()
    if len(geometries) != 302:
        raise RuntimeError(f"expected 302 competition counties in target states, got {len(geometries)}")
    if args.skip_download and MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        tiles = manifest["tiles"]
        downloaded = tiles
    else:
        tiles, query_url = query_tiles(geometries)
        print(f"USGS query returned {len(tiles)} unique 1-degree tiles", flush=True)
        downloaded = download_tiles(tiles, args.workers)
        manifest = {
            "source": "USGS 3DEP / TNM Access",
            "dataset": "National Elevation Dataset (NED) 1 arc-second",
            "query_url": query_url,
            "resolution": "1 arc-second (~30 m)",
            "vertical_units": "meters, NAVD88 in conterminous US",
            "terrain_ruggedness_definition": "mean valid 8-neighbour absolute elevation difference in meters",
            "tiles": downloaded,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # A manifest from a previous run may contain relative paths; normalize them.
    for item in downloaded:
        item["local_path"] = str(Path(item["local_path"]))
        if not Path(item["local_path"]).exists():
            item["local_path"] = str(DEM_DIR / Path(urllib.parse.urlparse(item["url"]).path).name)
    result = county_stats(geometries, downloaded)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"saved {OUTPUT} rows={len(result)}", flush=True)
    print(result.describe(include="all").to_string(), flush=True)


if __name__ == "__main__":
    main()
