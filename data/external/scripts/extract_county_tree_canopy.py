"""Extract county-level NLCD Tree Canopy Cover statistics from a GeoTIFF or ZIP.

Example:
    python data/extract_county_tree_canopy.py \
        --raster data/nlcd_2025_tree_canopy_cover_conus.zip

The output mean is a percentage in the 0-100 range. This script is for the
continuous Tree Canopy Cover product, not categorical NLCD Land Cover data.
"""

import argparse
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window


TARGET_STATEFPS = {"18", "39", "42", "54"}
TCC_MIN_VALUE = 0
TCC_MAX_VALUE = 100
TCC_NON_PROCESSING_VALUE = 254
TCC_BACKGROUND_VALUE = 255
DATA_DIR = Path(__file__).resolve().parent


def raster_source(raster_path: Path, tif_member: str | None) -> str:
    """Return a rasterio-readable path, including a TIFF stored in a ZIP."""
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    if raster_path.suffix.lower() not in {".zip"}:
        return str(raster_path)

    with ZipFile(raster_path) as archive:
        tif_members = [
            member
            for member in archive.namelist()
            if member.lower().endswith((".tif", ".tiff"))
        ]

    if tif_member:
        if tif_member not in tif_members:
            raise ValueError(
                f"--tif-member '{tif_member}' is not a TIFF in {raster_path.name}. "
                f"Found: {tif_members}"
            )
    elif len(tif_members) == 1:
        tif_member = tif_members[0]
    elif not tif_members:
        raise ValueError(f"No .tif or .tiff found in ZIP: {raster_path}")
    else:
        raise ValueError(
            "ZIP contains multiple TIFFs. Choose one with --tif-member. "
            f"Found: {tif_members}"
        )

    return f"zip://{raster_path.resolve().as_posix()}!{tif_member}"


def sampled_range(src: rasterio.DatasetReader) -> tuple[float, float, int]:
    """Validate valid TCC values while excluding the product's special codes."""
    size = min(512, src.width, src.height)
    row_starts = np.linspace(0, max(0, src.height - size), 5, dtype=int)
    col_starts = np.linspace(0, max(0, src.width - size), 5, dtype=int)
    values = []

    for row_start in row_starts:
        for col_start in col_starts:
            sample = src.read(1, window=Window(col_start, row_start, size, size))
            sample = sample[
                (sample >= TCC_MIN_VALUE)
                & (sample <= TCC_MAX_VALUE)
            ]
            if sample.size:
                values.append(sample)

    if not values:
        raise ValueError("No valid 0-100 TCC pixels found in sampled raster windows.")

    sample_values = np.concatenate(values)
    return float(sample_values.min()), float(sample_values.max()), int(sample_values.size)


def county_stats(src: rasterio.DatasetReader, geometry: object) -> dict[str, float | int | None]:
    """Calculate statistics excluding TCC's 254 non-processing and 255 background codes."""
    pixels, _ = mask(src, [geometry], crop=True, filled=False)
    pixels = pixels[0]
    inside_county = ~np.ma.getmaskarray(pixels)
    non_processing = inside_county & (pixels.data == TCC_NON_PROCESSING_VALUE)
    valid = inside_county & (pixels.data >= TCC_MIN_VALUE) & (pixels.data <= TCC_MAX_VALUE)
    values = pixels.data[valid].astype(np.float64, copy=False)

    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
            "non_processing_count": int(non_processing.sum()),
        }

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.size),
        "non_processing_count": int(non_processing.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate county-level NLCD Tree Canopy Cover statistics."
    )
    parser.add_argument(
        "--raster",
        required=True,
        help="2025 NLCD Tree Canopy Cover GeoTIFF or its containing ZIP file.",
    )
    parser.add_argument(
        "--tif-member",
        help="TIFF member path inside the ZIP, required only when it contains multiple TIFFs.",
    )
    parser.add_argument(
        "--counties",
        default=str(DATA_DIR / "tl_2025_us_county.shp"),
        help="TIGER/Line county shapefile (default: data/tl_2025_us_county.shp).",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "county_tree_canopy_2025.csv"),
        help="Output CSV path (default: data/county_tree_canopy_2025.csv).",
    )
    args = parser.parse_args()

    raster_path = Path(args.raster)
    source = raster_source(raster_path, args.tif_member)

    print(f"[1/4] Reading canopy raster: {raster_path}")
    with rasterio.open(source) as src:
        canopy_crs = src.crs
        nodata = src.nodata
        value_min, value_max, sample_count = sampled_range(src)
        print(f"  CRS: {canopy_crs}")
        print(f"  NoData: {nodata}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Valid TCC sample range: {value_min:g} to {value_max:g}")
        print(
            "  Special codes: 254 = non-processing area; "
            "255 = background (both excluded from statistics)"
        )

    if value_min < TCC_MIN_VALUE or value_max > TCC_MAX_VALUE or sample_count == 0:
        raise ValueError(
            "This raster is not a 0-100 Tree Canopy Cover percentage product. "
            "Check that the downloaded file is NLCD Tree Canopy Cover."
        )

    print(f"[2/4] Reading county boundaries: {args.counties}")
    counties = gpd.read_file(args.counties)
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties = counties[counties["STATEFP"].isin(TARGET_STATEFPS)].copy()
    if len(counties) != 302:
        raise ValueError(f"Expected 302 counties for IN/OH/PA/WV; found {len(counties)}.")
    if counties.crs != canopy_crs:
        print(f"  Reprojecting: {counties.crs} -> {canopy_crs}")
        counties = counties.to_crs(canopy_crs)
    print(f"  Counties selected: {len(counties)}")

    print("[3/4] Calculating county canopy statistics...")
    print("  Excluding TCC special values 254 (non-processing) and 255 (background).")

    rows = []
    with rasterio.open(source) as src:
        for county in counties.itertuples(index=False):
            stat = county_stats(src, county.geometry)
            count = int(stat.get("count") or 0)
            non_processing_count = int(stat["non_processing_count"])
            rows.append(
                {
                    "fipsCode": str(county.GEOID).zfill(5),
                    "county_name": county.NAME,
                    "tree_canopy_pct": round(float(stat["mean"]), 2) if count else None,
                    "tree_canopy_std": round(float(stat["std"]), 2) if count else None,
                    "tree_canopy_min_pct": round(float(stat["min"]), 2) if count else None,
                    "tree_canopy_max_pct": round(float(stat["max"]), 2) if count else None,
                    "valid_pixels": count,
                    "non_processing_pixels": non_processing_count,
                    "valid_pixel_pct": round(
                        100.0 * count / (count + non_processing_count), 2
                    ) if count + non_processing_count else None,
                }
            )

    output = pd.DataFrame(rows).sort_values("fipsCode")
    if output["tree_canopy_pct"].isna().any():
        missing = output.loc[output["tree_canopy_pct"].isna(), "fipsCode"].tolist()
        raise ValueError(f"No valid canopy pixels for FIPS: {missing}")
    if output["tree_canopy_min_pct"].min() < 0 or output["tree_canopy_max_pct"].max() > 100:
        raise ValueError("County statistics fall outside 0-100%; aborting output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"[4/4] Wrote {len(output)} rows: {output_path}")
    print(
        "  County mean tree canopy cover: "
        f"{output['tree_canopy_pct'].mean():.2f}% "
        f"({output['tree_canopy_pct'].min():.2f}% to {output['tree_canopy_pct'].max():.2f}%)"
    )


if __name__ == "__main__":
    main()
