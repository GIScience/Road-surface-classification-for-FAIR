import json
import logging
import math
import shutil
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from PIL import Image
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window
from shapely import STRtree, box
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

DATASET_CONFIG_FILENAME = "config.json"
ROADS_FILENAME = "roads.parquet"
PATCHES_FILENAME = "patches.parquet"


@dataclass
class PatchConfig:
    patch_size_m: float = 15.0
    patch_size_px: int = 256
    meters_per_patch: float = 20.0
    min_patches_per_road: int = 1
    max_patches_per_road: int = 5
    n_bags: int = 10
    endpoint_margin_fraction: float = 0.1
    junction_clearance_m: float = 7.5
    lateral_jitter_m: float = 1.5
    bearing_jitter_deg: float = 10.0
    max_nodata_fraction: float = 0.1
    max_attempts_per_patch: int = 10
    fill_tolerance: int = 1
    junction_exclude_highway_values: list[str] = field(default_factory=list)


def missing_label_mask(labels: pd.Series) -> pd.Series:
    return labels.isna() | labels.apply(lambda v: isinstance(v, str) and v.strip() == "")


def load_lines(vector_path, crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(vector_path).to_crs(crs)
    gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
    return gdf.explode(index_parts=False).rename_axis("source_row").reset_index()


def roads_to_cut(lines: gpd.GeoDataFrame, label_column) -> gpd.GeoDataFrame:
    if label_column is None:
        roads = lines.assign(label=None)
    else:
        roads = lines[~missing_label_mask(lines[label_column])].rename(columns={label_column: "label"})
    return roads[["source_row", "label", "geometry"]].reset_index(drop=True)


def junction_network(lines: gpd.GeoDataFrame, exclude_highway_values: list[str]) -> gpd.GeoSeries:
    if "highway" not in lines:
        return lines.geometry
    return lines.geometry[~lines["highway"].isin(exclude_highway_values)]


def scene_utm_crs(raster) -> rasterio.crs.CRS:
    footprint = gpd.GeoSeries([box(*raster.bounds)], crs=raster.crs)
    return rasterio.crs.CRS.from_user_input(footprint.estimate_utm_crs())


def patches_for_road(length_m: float, config: PatchConfig) -> int:
    return int(min(max(length_m // config.meters_per_patch, config.min_patches_per_road), config.max_patches_per_road))


def junction_distances(lines, network, radius: float) -> list[list[float]]:
    network = np.asarray(network)
    tree = STRtree(network)
    per_line = []
    for line in lines:
        others = [other for other in network[tree.query(line, predicate="dwithin", distance=radius)] if not other.equals(line)]
        # Crossings, plus other roads' endpoints beside this one: an unsnapped endpoint is almost always
        # a junction the data never snapped.
        contacts = shapely.get_parts(shapely.extract_unique_points(
            [line.intersection(other) for other in others] + [other.boundary for other in others]
        ))
        near = contacts[shapely.distance(line, contacts) <= radius]
        per_line.append(sorted(set(shapely.line_locate_point(line, near).tolist())))
    return per_line


def allowed_distances(length: float, junctions, margin: float, clearance: float, step: float = 0.5) -> np.ndarray:
    spots = np.linspace(margin, length - margin, int((length - 2 * margin) / step) + 1)
    clear = np.all(np.abs(spots[:, None] - np.asarray(junctions)[None, :]) >= clearance, axis=1)
    return spots[clear] if clear.any() else spots


def line_bearing_deg(line, distance: float) -> float:
    eps = min(1.0, line.length / 4)
    p0 = line.interpolate(max(distance - eps, 0.0))
    p1 = line.interpolate(min(distance + eps, line.length))
    return math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x))


def patch_corners(x: float, y: float, bearing_deg: float, size_m: float) -> np.ndarray:
    theta = math.radians(bearing_deg)
    ahead = np.array([math.cos(theta), math.sin(theta)]) * size_m / 2
    right = np.array([math.sin(theta), -math.cos(theta)]) * size_m / 2
    center = np.array([x, y])
    # The road runs up the patch. Order: top left, bottom left, bottom right, top right, as PIL's QUAD wants them.
    return np.array([center + ahead - right, center - ahead - right, center - ahead + right, center + ahead + right])


def read_patch(raster, corners: np.ndarray, size_px: int, crs) -> Image.Image:
    xs, ys = transform_coords(crs, raster.crs, corners[:, 0], corners[:, 1])
    cols, rows = ~raster.transform * (np.asarray(xs), np.asarray(ys))
    margin = 2  # pixels either side, so bilinear has real data to interpolate right at the edge
    col0, row0 = math.floor(cols.min()) - margin, math.floor(rows.min()) - margin
    width, height = math.ceil(cols.max()) + margin - col0, math.ceil(rows.max()) + margin - row0
    window = raster.read(window=Window(col0, row0, width, height), boundless=True, fill_value=0)
    quad = np.column_stack([cols - col0, rows - row0]).ravel().tolist()
    return Image.fromarray(window.transpose(1, 2, 0)).transform((size_px, size_px), Image.QUAD, quad, Image.BILINEAR)


def nodata_fraction(image: Image.Image, fill_tolerance: int) -> float:
    # Off-raster pixels read as 0 and most scenes declare no nodata at all, so blank fill (near-white or
    # near-black in every band) is what counts as invalid.
    pixels = np.asarray(image)
    return float(((pixels >= 255 - fill_tolerance).all(-1) | (pixels <= fill_tolerance).all(-1)).mean())


def cut_patch(raster, crs, line, distance: float, config: PatchConfig, rng):
    on_road = line.interpolate(distance)
    bearing = line_bearing_deg(line, distance)
    theta = math.radians(bearing)
    lateral = rng.uniform(-config.lateral_jitter_m, config.lateral_jitter_m)
    rotation = rng.uniform(-config.bearing_jitter_deg, config.bearing_jitter_deg)
    corners = patch_corners(on_road.x + lateral * math.sin(theta), on_road.y - lateral * math.cos(theta),
                            bearing + rotation, config.patch_size_m)
    return read_patch(raster, corners, config.patch_size_px, crs), Polygon(corners)


def cut_bag(raster, crs, line, junctions, n_patches: int, offset: float, config: PatchConfig, rng) -> list:
    margin = line.length * config.endpoint_margin_fraction
    allowed = allowed_distances(line.length, junctions, margin, config.junction_clearance_m)
    cell = len(allowed) / n_patches
    bag = []
    for slot in range(n_patches):
        index = (slot + offset) * cell
        for _ in range(config.max_attempts_per_patch):
            distance = float(allowed[int(index)])
            image, footprint = cut_patch(raster, crs, line, distance, config, rng)
            if nodata_fraction(image, config.fill_tolerance) <= config.max_nodata_fraction:
                bag.append((slot, distance, image, footprint))
                break
            index = rng.uniform(slot * cell, (slot + 1) * cell)
    return bag


def materialize_scene(scene_id: str, raster_path, vector_path, scene_dir, label_column, config: PatchConfig, seed=0) -> None:
    rng = np.random.default_rng([seed, zlib.crc32(scene_id.encode())])
    shutil.rmtree(scene_dir, ignore_errors=True)
    scene_dir.mkdir(parents=True)

    with rasterio.open(raster_path) as raster:
        crs = scene_utm_crs(raster)
        lines = load_lines(vector_path, crs)
        roads = roads_to_cut(lines, label_column)
        roads["n_patches"] = [patches_for_road(line.length, config) for line in roads.geometry]
        network = junction_network(lines, config.junction_exclude_highway_values)
        junctions = junction_distances(roads.geometry, network, config.patch_size_m / 2)

        patches = []
        for road_id, (line, road_junctions, n_patches) in enumerate(zip(roads.geometry, junctions, roads["n_patches"])):
            for bag in range(config.n_bags):
                # Each bag's comb sits at its own offset in the cells, so the bags spread evenly along the road.
                offset = (bag + 0.5) / config.n_bags
                for slot, distance, image, footprint in cut_bag(raster, crs, line, road_junctions, n_patches, offset, config, rng):
                    path = f"{road_id:04d}/bag{bag:02d}/slot{slot}.png"
                    (scene_dir / path).parent.mkdir(parents=True, exist_ok=True)
                    image.save(scene_dir / path)
                    patches.append((road_id, bag, slot, path, distance, footprint))

    roads.rename_axis("road_id").reset_index().to_parquet(scene_dir / ROADS_FILENAME)
    columns = ["road_id", "bag", "slot", "path", "distance_m", "geometry"]
    gpd.GeoDataFrame(pd.DataFrame(patches, columns=columns), crs=crs).to_parquet(scene_dir / PATCHES_FILENAME)


def materialize_dataset(scenes: list[tuple], dataset_dir, label_column, config: PatchConfig, seed=0) -> None:
    dataset_dir = Path(dataset_dir)
    settings = dict(asdict(config), label_column=label_column, seed=seed)
    config_path = dataset_dir / DATASET_CONFIG_FILENAME
    stored = json.loads(config_path.read_text()) if config_path.exists() else settings
    changed = sorted(key for key in settings if stored.get(key) != settings[key])
    if changed:
        raise ValueError(f"{dataset_dir} was made with different {', '.join(changed)}; use a new dataset_dir for these settings")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(settings, indent=2))

    for scene_id, raster_path, vector_path in scenes:
        if (dataset_dir / scene_id / PATCHES_FILENAME).exists():
            logger.info("%s: already materialized, skipping", scene_id)
            continue
        logger.info("%s: materializing ...", scene_id)
        materialize_scene(scene_id, raster_path, vector_path, dataset_dir / scene_id, label_column, config, seed)
