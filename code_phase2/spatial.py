"""Read the supplied county geo table and construct a spatial kNN graph.

The downloaded file is a DBF accompanying the shapefile. Reading the small
DBF directly avoids making geopandas/pyshp a runtime dependency.
"""

import struct
from pathlib import Path

import numpy as np


def read_dbf_points(path: Path):
    raw = Path(path).read_bytes()
    n_records = struct.unpack_from("<I", raw, 4)[0]
    header_len = struct.unpack_from("<H", raw, 8)[0]
    record_len = struct.unpack_from("<H", raw, 10)[0]
    fields = []
    for pos in range(32, header_len - 1, 32):
        name = raw[pos:pos + 11].split(b"\x00", 1)[0].decode("ascii", "ignore").strip()
        if not name:
            continue
        fields.append((name, chr(raw[pos + 11]), raw[pos + 16]))

    offsets = {}
    cursor = 1
    for name, kind, width in fields:
        offsets[name] = (cursor, kind, width)
        cursor += width

    points = {}
    for i in range(n_records):
        rec = raw[header_len + i * record_len: header_len + (i + 1) * record_len]
        if not rec or rec[0:1] == b"*":
            continue

        def value(name):
            start, kind, width = offsets[name]
            text = rec[start:start + width].decode("ascii", "ignore").strip()
            if kind in "NF" and text:
                return float(text)
            return text

        fips = str(value("FIPS")).strip().zfill(5)
        try:
            lon = float(value("LON"))
            lat = float(value("LAT"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(lat) and np.isfinite(lon):
            points[fips] = (lat, lon)
    return points


def build_knn_graph(fips_codes, dbf_path, k=8):
    """Return node coordinates and directed edge_index [2, E]."""
    fips_codes = [str(x).zfill(5) for x in fips_codes]
    points = read_dbf_points(dbf_path)
    missing = [f for f in fips_codes if f not in points]
    if missing:
        raise ValueError(f"Missing coordinates for {len(missing)} counties, e.g. {missing[:5]}")
    coords = np.asarray([points[f] for f in fips_codes], dtype=np.float32)
    # Euclidean distance in degree coordinates is adequate at this compact
    # four-state scale; latitude/longitude are also supplied as node inputs.
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    edges = {(i, i) for i in range(len(fips_codes))}
    k = min(int(k), len(fips_codes) - 1)
    for i in range(len(fips_codes)):
        nearest = np.argsort(distance[i])[1:k + 1]
        for j in nearest:
            edges.add((i, int(j)))
            edges.add((int(j), i))
    edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    return coords, edge_index
