"""Read the supplied county geo table and construct a spatial kNN graph.

The downloaded file is a DBF accompanying the shapefile. Reading the small
DBF directly avoids making geopandas/pyshp a runtime dependency.
"""

import struct
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


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


def read_dbf_fips_in_order(path: Path):
    """Return DBF FIPS values in the same order as shapefile records."""
    raw = Path(path).read_bytes()
    n_records = struct.unpack_from("<I", raw, 4)[0]
    header_len = struct.unpack_from("<H", raw, 8)[0]
    record_len = struct.unpack_from("<H", raw, 10)[0]
    fields = []
    for pos in range(32, header_len - 1, 32):
        name = raw[pos:pos + 11].split(b"\x00", 1)[0].decode("ascii", "ignore").strip()
        if name:
            fields.append((name, raw[pos + 16]))
    cursor = 1
    offsets = {}
    for name, width in fields:
        offsets[name] = (cursor, width)
        cursor += width
    start, width = offsets["FIPS"]
    out = []
    for i in range(n_records):
        rec = raw[header_len + i * record_len:header_len + (i + 1) * record_len]
        text = rec[start:start + width].decode("ascii", "ignore").strip()
        out.append(text.zfill(5))
    return out


def read_shp_polygons(path: Path):
    """Read polygon records from a simple ESRI Polygon shapefile.

    This intentionally avoids geopandas/fiona. The supplied county file is a
    standard shape type 5 file; each returned geometry is aligned to the DBF
    record order.
    """
    raw = Path(path).read_bytes()
    offset = 100
    geometries = []
    while offset + 8 <= len(raw):
        _, content_words = struct.unpack_from(">2i", raw, offset)
        offset += 8
        size = content_words * 2
        content = raw[offset:offset + size]
        offset += size
        if len(content) < 4:
            geometries.append(None)
            continue
        shape_type = struct.unpack_from("<i", content, 0)[0]
        if shape_type == 0:
            geometries.append(None)
            continue
        if shape_type != 5:
            raise ValueError(f"Unsupported county shapefile shape type: {shape_type}")
        num_parts = struct.unpack_from("<i", content, 36)[0]
        num_points = struct.unpack_from("<i", content, 40)[0]
        parts = struct.unpack_from(f"<{num_parts}i", content, 44)
        point_start = 44 + 4 * num_parts
        points = [struct.unpack_from("<2d", content, point_start + 16 * i)
                  for i in range(num_points)]
        rings = []
        for part_idx, begin in enumerate(parts):
            end = parts[part_idx + 1] if part_idx + 1 < num_parts else num_points
            ring = points[begin:end]
            if len(ring) >= 4:
                rings.append(Polygon(ring).buffer(0))
        geometries.append(unary_union(rings) if rings else None)
    return geometries


def build_spatial_graph(fips_codes, dbf_path, shp_path, k=8):
    """Build a symmetric graph from kNN plus exact shared county borders."""
    fips_codes = [str(x).zfill(5) for x in fips_codes]
    points = read_dbf_points(dbf_path)
    missing = [f for f in fips_codes if f not in points]
    if missing:
        raise ValueError(f"Missing coordinates for {len(missing)} counties, e.g. {missing[:5]}")
    coords = np.asarray([points[f] for f in fips_codes], dtype=np.float32)
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    edges = {(i, i) for i in range(len(fips_codes))}
    edge_types = {(i, i): 0.0 for i in range(len(fips_codes))}
    k = min(int(k), len(fips_codes) - 1)
    for i in range(len(fips_codes)):
        for j in np.argsort(distance[i])[1:k + 1]:
            edges.add((i, int(j)))
            edges.add((int(j), i))
            edge_types[(i, int(j))] = max(edge_types.get((i, int(j)), 0.0), 0.0)
            edge_types[(int(j), i)] = max(edge_types.get((int(j), i), 0.0), 0.0)

    # Exact shared-border edges from the existing shapefile. Only the 302
    # competition counties are materialized, avoiding a full-US O(n^2) check.
    dbf_fips = read_dbf_fips_in_order(dbf_path)
    shape_by_fips = {}
    for fips, geometry in zip(dbf_fips, read_shp_polygons(shp_path)):
        if fips in set(fips_codes) and geometry is not None:
            shape_by_fips[fips] = geometry
    for i, fi in enumerate(fips_codes):
        gi = shape_by_fips.get(fi)
        if gi is None:
            continue
        for j in range(i + 1, len(fips_codes)):
            gj = shape_by_fips.get(fips_codes[j])
            if gj is None or not gi.bounds or not gj.bounds:
                continue
            if (gi.bounds[2] < gj.bounds[0] - 1e-7 or
                    gj.bounds[2] < gi.bounds[0] - 1e-7 or
                    gi.bounds[3] < gj.bounds[1] - 1e-7 or
                    gj.bounds[3] < gi.bounds[1] - 1e-7):
                continue
            shared = gi.boundary.intersection(gj.boundary)
            if not shared.is_empty and shared.length > 1e-6:
                edges.add((i, j))
                edges.add((j, i))
                edge_types[(i, j)] = 1.0
                edge_types[(j, i)] = 1.0
    edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    attrs = []
    lat_scale = np.cos(np.deg2rad(float(np.mean(coords[:, 0]))))
    for src, dst in edge_index.T:
        dlat = float(coords[dst, 0] - coords[src, 0])
        dlon = float(coords[dst, 1] - coords[src, 1]) * lat_scale
        distance_km = 111.0 * np.sqrt(dlat * dlat + dlon * dlon)
        bearing = np.arctan2(dlon, dlat) if distance_km > 0 else 0.0
        attrs.append([
            min(distance_km / 500.0, 2.0),
            np.sin(bearing),
            np.cos(bearing),
            edge_types.get((int(src), int(dst)), 0.0),
        ])
    return coords, edge_index, np.asarray(attrs, dtype=np.float32)


def build_knn_graph(fips_codes, dbf_path, k=8):
    """Return node coordinates and directed edge_index [2, E]."""
    # Backward-compatible centroid-only graph for external callers.
    fips_codes = [str(x).zfill(5) for x in fips_codes]
    points = read_dbf_points(dbf_path)
    coords = np.asarray([points[f] for f in fips_codes], dtype=np.float32)
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    edges = {(i, i) for i in range(len(fips_codes))}
    for i in range(len(fips_codes)):
        for j in np.argsort(distance[i])[1:min(int(k), len(fips_codes) - 1) + 1]:
            edges.add((i, int(j))); edges.add((int(j), i))
    return coords, np.asarray(sorted(edges), dtype=np.int64).T
