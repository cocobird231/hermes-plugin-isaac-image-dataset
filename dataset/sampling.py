"""Bounded schema-2 semantic-face sampling; CPU/OpenUSD only, never mutates a stage.

Geometry checks use the authored static triangle meshes, not NuRec radiance/opacity.
Ground support is exact at the centre; clearance is a conservative square column,
not a vehicle footprint, continuous free-space guarantee or a visibility certificate.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from pxr import Ar, Sdf, Usd, UsdGeom, UsdUtils

from .pose_math import pose_from_angles
from .quality import DEFAULT_RENDER_QUALITY, normalize_render_quality

_ALLOWED_CLASSES = {"road_surface", "drivable_area", "lane"}
_DEFAULTS = {
    "classes": ["road_surface"], "n": 32, "seed": 42, "camera_count": 1, "sensor_height_m": 1.5,
    "render_quality": DEFAULT_RENDER_QUALITY,
    "yaw_deg": [0, 360], "pitch_deg": [-5, 10], "roll_deg": [-5, 5],
    "min_distance_m": 3., "group_distance_m": 2., "max_slope_deg": 15.,
    "clearance_m": .15, "max_attempts": 10000,
    "camera": {"resolution": [1280, 720], "horizontal_fov_deg": 90.,
               "clipping_range": [.1, 2000.]},
    "split_ratios": [4, 2, 2],
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _number(value, name, lower=None, upper=None):
    try:
        finite = math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite:
        raise ValueError(f"{name} must be finite numeric, not boolean")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} outside supported bounds")
    return float(value)


def normalize_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    unknown = set(config) - set(_DEFAULTS) - {"scene_path", "scene_sha256"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    result = json.loads(json.dumps(_DEFAULTS))
    result.update(config)
    if not isinstance(result.get("scene_path"), str) or not result["scene_path"].strip():
        raise ValueError("scene_path is required")
    scene_path = Path(result["scene_path"]).expanduser()
    if not scene_path.is_absolute():
        raise ValueError("scene_path must be an absolute host path")
    result["scene_path"] = str(scene_path.resolve(strict=True))
    if not Path(result["scene_path"]).is_file():
        raise ValueError("scene_path must be a file")
    if Path(result["scene_path"]).suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError("Unsupported scene extension")
    expected = result.get("scene_sha256")
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected)):
        raise ValueError("scene_sha256 must be lowercase SHA256")
    for name, lower, upper in (("n", 1, 10000), ("seed", 0, 2**63-1), ("camera_count", 1, 8), ("max_attempts", 1, 1000000)):
        if type(result[name]) is not int or not lower <= result[name] <= upper:
            raise ValueError(f"{name} must be integer in [{lower}, {upper}]")
    result["render_quality"] = normalize_render_quality(result["render_quality"])
    if result["max_attempts"] < result["n"]:
        raise ValueError("max_attempts must be >= n")
    classes = result["classes"]
    if (not isinstance(classes, list) or not classes or any(not isinstance(c, str) for c in classes)
            or len(classes) != len(set(classes)) or set(classes) - _ALLOWED_CLASSES):
        raise ValueError("classes must be distinct eligible road_surface/drivable_area/lane labels")
    for name, lo, hi in (("sensor_height_m", .05, 100.), ("min_distance_m", 0., 10000.),
                         ("group_distance_m", 0., 10000.), ("max_slope_deg", 0., 89.),
                         ("clearance_m", .001, 10.)):
        result[name] = _number(result[name], name, lo, hi)
    if result["sensor_height_m"] <= result["clearance_m"]:
        raise ValueError("sensor_height_m must exceed clearance_m")
    ground_band = max(.10, result["clearance_m"] * math.sqrt(2) * math.tan(math.radians(result["max_slope_deg"])) + .02)
    if result["sensor_height_m"] + result["clearance_m"] <= ground_band:
        raise ValueError("Sensor height is too low for the configured ground clearance band")
    for name, lo, hi in (("yaw_deg", -360., 360.), ("pitch_deg", -89., 89.), ("roll_deg", -180., 180.)):
        pair = result[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name} requires two bounds")
        result[name] = [_number(v, name, lo, hi) for v in pair]
        if result[name][0] > result[name][1]:
            raise ValueError(f"{name} bounds must be increasing")
    ratios = result["split_ratios"]
    if not isinstance(ratios, (list, tuple)) or len(ratios) != 3:
        raise ValueError("split_ratios must contain train/validation/test weights")
    result["split_ratios"] = [_number(v, "split_ratios", 0.) for v in ratios]
    if not 0 < sum(result["split_ratios"]) < float("inf"):
        raise ValueError("split_ratios sum must be positive and finite")
    camera = result["camera"]
    if (not isinstance(camera, dict)
            or set(camera) - {"resolution", "horizontal_fov_deg", "clipping_range", "template_prim"}):
        raise ValueError("Only pinhole resolution/horizontal_fov_deg/clipping_range/template_prim are supported")
    explicit_clip = "clipping_range" in camera
    camera = {**_DEFAULTS["camera"], **camera}
    if "template_prim" in camera:
        template = camera["template_prim"]
        if (not isinstance(template, str) or not template.startswith("/")
                or not Sdf.Path.IsValidPathString(template)
                or not Sdf.Path(template).IsPrimPath()):
            raise ValueError("template_prim must be an absolute USD prim path")
    if (not isinstance(camera["resolution"], (list, tuple)) or len(camera["resolution"]) != 2
            or any(type(v) is not int or not 16 <= v <= 4096 for v in camera["resolution"])):
        raise ValueError("Camera resolution requires two integers in [16,4096]")
    camera["resolution"] = list(camera["resolution"])
    camera["horizontal_fov_deg"] = _number(camera["horizontal_fov_deg"], "horizontal_fov_deg", 2., 175.)
    clip = camera["clipping_range"]
    if not isinstance(clip, (list, tuple)) or len(clip) != 2:
        raise ValueError("clipping_range requires two positive bounds")
    camera["clipping_range"] = [_number(v, "clipping_range", .000001) for v in clip]
    if camera["clipping_range"][0] >= camera["clipping_range"][1]:
        raise ValueError("clipping_range must increase")
    if camera.get("template_prim") and not explicit_clip:
        # Preserve absence so the renderer inherits the template clipping range.
        del camera["clipping_range"]
    result["camera"] = camera
    return result


def _filesystem_dependency(identifier):
    if Ar.IsPackageRelativePath(identifier):
        identifier = Ar.SplitPackageRelativePathOuter(identifier)[0]
    path = Path(identifier)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Nonlocal or missing scene dependency: {identifier}")
    return str(path.resolve())


def _load_scene(config):
    path = config["scene_path"]
    digest = sha256(path)
    if config.get("scene_sha256") and digest != config["scene_sha256"]:
        raise ValueError("Scene SHA256 mismatch")
    config["scene_sha256"] = digest
    stage = Usd.Stage.Open(path, load=Usd.Stage.LoadAll)
    if not stage or stage.GetCompositionErrors():
        raise ValueError("Scene must compose without errors")
    if config["camera"].get("template_prim"):
        from .render_worker import configured_camera
        configured_camera(stage, config["camera"])
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("Only metre Z-up stages supported")
    scope = stage.GetPrimAtPath("/DatasetSemantics")
    if not scope or not scope.IsA(UsdGeom.Scope):
        raise ValueError("Schema-2 /DatasetSemantics Scope is required")
    try:
        annotation = json.loads(scope.GetAttribute("dataset:annotationJson").Get())
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid semantic annotation JSON") from exc
    if type(annotation.get("schema_version")) is not int or annotation["schema_version"] != 2:
        raise ValueError("Only schema-2 source_mesh_faces semantics supported")
    root = stage.GetRootLayer()
    if len(root.subLayerPaths) != 1:
        raise ValueError("Semantic overlay must have exactly one source sublayer")
    source = Sdf.ComputeAssetPathRelativeToLayer(root, root.subLayerPaths[0])
    source = _filesystem_dependency(source)
    if sha256(source) != annotation.get("source_sha256"):
        raise ValueError("Actual source sublayer SHA does not match semantic annotation")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(path)
    if unresolved:
        raise ValueError(f"Unresolved scene dependencies: {unresolved}")
    paths = {_filesystem_dependency(layer.realPath or layer.identifier) for layer in layers}
    paths.update(_filesystem_dependency(str(asset)) for asset in assets)
    dependencies = {p: sha256(p) for p in sorted(paths)}
    if source not in dependencies:
        raise ValueError("Source sublayer not present in resolved dependencies")
    meshes = []
    for prim in stage.Traverse():
        if prim.IsInstance() or prim.IsInstanceProxy():
            raise ValueError("Instanced scene geometry is unsupported")
        if prim.IsA(UsdGeom.PointInstancer):
            raise ValueError("PointInstancer geometry is unsupported")
        if not prim.IsA(UsdGeom.Mesh):
            if prim.IsA(UsdGeom.Gprim):
                raise ValueError(f"Non-mesh geometric primitive unsupported: {prim.GetPath()}")
            continue
        mesh = UsdGeom.Mesh(prim)
        attributes = [mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(),
                      mesh.GetFaceVertexIndicesAttr(), mesh.GetHoleIndicesAttr()]
        if any(root.GetAttributeAtPath(attr.GetPath()) for attr in attributes):
            raise ValueError("Semantic overlay must not override source mesh geometry")
        if any(attr.GetNumTimeSamples() for attr in attributes):
            raise ValueError("Time-sampled geometry unsupported")
        if len(mesh.GetHoleIndicesAttr().Get() or []):
            raise ValueError("Authored holeIndices must be triangulated/removed before sampling")
        ancestor = prim
        while ancestor and ancestor.GetPath() != Sdf.Path.absoluteRootPath:
            xform_attrs = [a for a in ancestor.GetAttributes() if a.GetName().startswith("xformOp:") or a.GetName() == "xformOpOrder"]
            if any(root.GetAttributeAtPath(a.GetPath()) for a in xform_attrs):
                raise ValueError("Semantic overlay must not override source transforms")
            if any(a.GetNumTimeSamples() for a in xform_attrs):
                raise ValueError("Animated transforms unsupported")
            ancestor = ancestor.GetParent()
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        transform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if (counts.ndim != 1 or len(counts) == 0 or len(counts) > 20_000_000
                or not np.all(counts == 3) or len(indices) != len(counts)*3
                or points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
                or np.any(indices < 0) or np.any(indices >= len(points))
                or not np.isfinite(transform).all() or not np.allclose(transform[:3, 3], 0)
                or not np.isclose(transform[3, 3], 1.)):
            raise ValueError(f"Finite static triangulated affine mesh required: {prim.GetPath()}")
        points = points @ transform[:3, :3] + transform[3, :3]
        indices = indices.reshape(-1, 3)
        lower = np.empty((len(indices), 3), dtype=np.float64)
        upper = np.empty_like(lower)
        for start in range(0, len(indices), 100000):
            tri = points[indices[start:start+100000]]
            lower[start:start+len(tri)] = tri.min(axis=1)
            upper[start:start+len(tri)] = tri.max(axis=1)
        meshes.append({"path": str(prim.GetPath()), "points": points, "indices": indices,
                       "lower": lower, "upper": upper})
    ground = next((m for m in meshes if m["path"] == annotation.get("source_ground_prim")), None)
    if ground is None or len(ground["indices"]) != annotation.get("source_face_count"):
        raise ValueError("Source ground mesh/face count mismatch")
    regions = annotation.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Nonempty semantic regions required")
    seen = np.zeros(len(ground["indices"]), dtype=bool)
    selected, region_indices, region_ids = [], [], []
    for number, region in enumerate(regions):
        rid = region.get("id")
        if not isinstance(rid, str) or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", rid):
            raise ValueError("Invalid region identifier")
        if rid in region_ids:
            raise ValueError("Duplicate region identifier")
        region_ids.append(rid)
        prim = stage.GetPrimAtPath("/DatasetSemantics/" + rid)
        if not prim or not prim.IsA(UsdGeom.Scope):
            raise ValueError("Region Scope missing")
        ids = np.asarray(prim.GetAttribute("dataset:sourceFaceIndices").Get())
        if (ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= len(seen)) or len(np.unique(ids)) != len(ids)):
            raise ValueError("Invalid source face indices")
        if np.any(seen[ids]):
            raise ValueError("Semantic regions overlap")
        seen[ids] = True
        eligible = prim.GetAttribute("dataset:samplingEligible").Get()
        label = region.get("class")
        subset_path = prim.GetAttribute("dataset:subsetPath").Get()
        subset = UsdGeom.Subset(stage.GetPrimAtPath(subset_path or ""))
        if (prim.GetAttribute("dataset:representation").Get() != "source_mesh_faces"
                or prim.GetAttribute("dataset:sourceGroundPrim").Get() != ground["path"]
                or list(prim.GetAttribute("semantics:labels:class").Get() or []) != [label]
                or type(eligible) is not bool or type(region.get("sampling_eligible")) is not bool
                or eligible != region.get("sampling_eligible")
                or len(ids) != region.get("face_count") or subset_path != region.get("subset_path")
                or not subset or subset.GetPrim().GetParent().GetPath() != Sdf.Path(ground["path"])
                or subset.GetFamilyNameAttr().Get() != "dataset_semantics_v2"
                or subset.GetElementTypeAttr().Get() != "face"
                or not np.array_equal(ids, np.asarray(subset.GetIndicesAttr().Get()))
                or list(subset.GetPrim().GetAttribute("semantics:labels:class").Get() or []) != [label]):
            raise ValueError(f"Semantic Scope/GeomSubset/annotation mismatch: {rid}")
        if eligible:
            if label not in _ALLOWED_CLASSES or region.get("geometry_support", {}).get("verified") is not True:
                raise ValueError("Eligible region lacks verified support contract")
            if label in config["classes"]:
                selected.append(ids.astype(np.int64))
                region_indices.append(np.full(len(ids), number, dtype=np.int32))
    if not selected:
        raise ValueError("No eligible source faces match requested classes")
    ids, owners = np.concatenate(selected), np.concatenate(region_indices)
    triangles = ground["points"][ground["indices"][ids]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    areas = np.abs(normals[:, 2]) / 2
    slope = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]) / np.maximum(lengths, 1e-30), 0., 1.)))
    keep = (areas > 1e-12) & (slope <= config["max_slope_deg"])
    if not keep.any():
        raise ValueError("No eligible nondegenerate faces satisfy max_slope_deg")
    metadata = {"path": path, "sha256": digest, "source_dependencies": dependencies,
                "source_ground_prim": ground["path"], "source_sha256": annotation["source_sha256"],
                "source_face_count": len(ground["indices"]), "meters_per_unit": 1.,
                "up_axis": "Z", "annotation_schema_version": 2}
    return metadata, meshes, ground, ids[keep], owners[keep], region_ids, triangles[keep], areas[keep], len(ids)


def _vertical_hits(triangles, xy):
    """Exact double-sided vertical intersections; vertical facets have no XY support."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    determinant = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    good = np.abs(determinant) > 1e-12
    safe = np.where(good, determinant, 1.)
    wa = ((b[:, 1]-c[:, 1])*(xy[0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(xy[1]-c[:, 1])) / safe
    wb = ((c[:, 1]-a[:, 1])*(xy[0]-c[:, 0]) + (a[:, 0]-c[:, 0])*(xy[1]-c[:, 1])) / safe
    wc = 1.-wa-wb
    good &= (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
    return good, wa*a[:, 2]+wb*b[:, 2]+wc*c[:, 2]


def triangle_box_intersects(triangles, lower, upper):
    """Triangle versus closed AABB via all 13 separating axes, including vertical facets."""
    tri = np.asarray(triangles, dtype=float) - (np.asarray(lower)+upper)/2
    half = (np.asarray(upper)-lower)/2
    if not len(tri):
        return np.zeros(0, dtype=bool)
    valid = np.all(tri.max(axis=1) >= -half-1e-10, axis=1) & np.all(tri.min(axis=1) <= half+1e-10, axis=1)
    edges = [tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1], tri[:, 0]-tri[:, 2]]
    axes = [np.cross(edges[0], edges[1])]
    axes.extend(np.cross(edge, axis) for edge in edges for axis in np.eye(3))
    for axis in axes:
        projections = np.einsum("nij,nj->ni", tri, axis)
        radius = np.abs(axis) @ half
        valid &= (projections.min(axis=1) <= radius+1e-10) & (projections.max(axis=1) >= -radius-1e-10)
    return valid


class _XYSpacingIndex:
    """Incremental exact XY spacing with bounded local buckets for accepted points."""
    def __init__(self, distance):
        self.distance = distance
        self.cells = {}

    def _cell(self, xy):
        return tuple(math.floor(float(v) / self.distance) for v in xy)

    def too_close(self, xy):
        if self.distance == 0:
            return False
        x, y = self._cell(xy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in self.cells.get((x+dx, y+dy), ()):
                    if math.hypot(px-float(xy[0]), py-float(xy[1])) < self.distance:
                        return True
        return False

    def add(self, xy):
        if self.distance:
            self.cells.setdefault(self._cell(xy), []).append((float(xy[0]), float(xy[1])))


def position_groups(samples, distance):
    """Exact inclusive XY connected components without constructing all pair edges.

    A half-radius cell is a clique. Union its occupants once, then test nearby
    cell pairs only until an edge is found. Already connected cell pairs need
    no further work; dense coincident samples remain linear in sample count.
    """
    distance = _number(distance, "group_distance_m", 0., 10000.)
    if not samples:
        return 0
    positions = np.asarray([s["pose_xyzw"][:2] for s in samples], dtype=float)
    if positions.shape != (len(samples), 2) or not np.isfinite(positions).all():
        raise ValueError("Grouping requires finite XY positions")
    threshold = distance + 1e-10
    cell_size = threshold / 2.
    parent = list(range(len(samples)))
    cells = {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    for i, xy in enumerate(positions):
        cell = tuple(math.floor(float(v) / cell_size) for v in xy)
        occupants = cells.setdefault(cell, [])
        if occupants:
            union(i, occupants[0])
        occupants.append(i)
    # Only one orientation of each pair; keys make ordering reproducible.
    for cell, occupants in cells.items():
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                other = (cell[0]+dx, cell[1]+dy)
                if other <= cell or other not in cells:
                    continue
                neighbors = cells[other]
                if root(occupants[0]) == root(neighbors[0]):
                    continue
                # Bounded temporary storage even for adversarial dense cells.
                connected = False
                for start in range(0, len(occupants), 128):
                    a = positions[occupants[start:start+128]]
                    for stop in range(0, len(neighbors), 128):
                        b = positions[neighbors[stop:stop+128]]
                        if np.any(np.hypot(a[:, None, 0]-b[None, :, 0],
                                           a[:, None, 1]-b[None, :, 1]) <= threshold):
                            union(occupants[0], neighbors[0])
                            connected = True
                            break
                    if connected:
                        break
    groups = {}
    for i, sample in enumerate(samples):
        key = root(i)
        groups.setdefault(key, f"group_{len(groups):06d}")
        sample["group_id"] = groups[key]
    return len(groups)


def generate_plan(config):
    """Return a complete plan or explicit bounded partial plan; no output/stage writes."""
    config = normalize_config(config)
    scene, meshes, ground, ids, owners, region_ids, triangles, areas, labelled_count = _load_scene(config)
    cdf = np.cumsum(areas)
    eligible = np.zeros(len(ground["indices"]), dtype=bool)
    eligible[ids] = True
    rng = np.random.default_rng(config["seed"])
    rejected, samples = Counter(), []
    spacing = _XYSpacingIndex(config["min_distance_m"])
    clearance = config["clearance_m"]
    ground_band = max(.10, clearance * math.sqrt(2) * math.tan(math.radians(config["max_slope_deg"])) + .02)
    for attempt in range(1, config["max_attempts"]+1):
        choice = int(np.searchsorted(cdf, rng.random()*cdf[-1], side="right"))
        uv = rng.random(2)
        if uv.sum() > 1:
            uv = 1.-uv
        triangle = triangles[choice]
        point = triangle[0] + uv[0]*(triangle[1]-triangle[0]) + uv[1]*(triangle[2]-triangle[0])
        if spacing.too_close(point[:2]):
            rejected["minimum_xy_spacing"] += 1
            continue
        # Re-intersect the selected actual source triangle, independent of barycentric generation.
        hit, z = _vertical_hits(triangle[None], point[:2])
        if not hit[0] or abs(z[0]-point[2]) > 1e-7:
            rejected["missing_ground_support"] += 1
            continue
        nearby = ((ground["lower"][:, 0] <= point[0]) & (ground["upper"][:, 0] >= point[0])
                  & (ground["lower"][:, 1] <= point[1]) & (ground["upper"][:, 1] >= point[1]) & eligible)
        near_ids = np.flatnonzero(nearby)
        hit, z = _vertical_hits(ground["points"][ground["indices"][near_ids]], point[:2])
        if np.any(hit & (np.abs(z-point[2]) > .15)):
            rejected["ambiguous_multiple_ground_layers"] += 1
            continue
        lower = point + [-clearance, -clearance, ground_band]
        upper = point + [clearance, clearance, config["sensor_height_m"]+clearance]
        blocked = False
        for mesh in meshes:
            near = np.all(mesh["upper"] >= lower, axis=1) & np.all(mesh["lower"] <= upper, axis=1)
            candidate = mesh["points"][mesh["indices"][near]]
            if triangle_box_intersects(candidate, lower, upper).any():
                blocked = True
                break
        if blocked:
            rejected["mesh_clearance_obstacle"] += 1
            continue
        angles = {name[:-4]: float(rng.uniform(*config[name])) for name in ("yaw_deg", "pitch_deg", "roll_deg")}
        position = point + [0, 0, config["sensor_height_m"]]
        pose, transform = pose_from_angles(position, angles["yaw"], angles["pitch"], angles["roll"])
        samples.append({"sample_id": f"{len(samples):06d}", "pose_xyzw": pose,
                        "T_world_camera": transform, "angles_deg": angles, "ground_z_m": float(point[2]),
                        "source_face_id": int(ids[choice]), "region_id": region_ids[int(owners[choice])]})
        spacing.add(point[:2])
        if len(samples) == config["n"]:
            break
    groups = position_groups(samples, config["group_distance_m"]) if samples else 0
    for path, expected in scene["source_dependencies"].items():
        if sha256(path) != expected:
            raise ValueError("Scene dependency changed during sampling: " + path)
    complete = len(samples) == config["n"]
    return {"schema_version": 1, "complete": complete, "config": config, "scene": scene, "samples": samples,
            "statistics": {"requested_count": config["n"], "accepted_count": len(samples), "attempts": attempt,
                           "rejected": dict(rejected), "eligible_labelled_face_count": labelled_count,
                           "slope_qualified_face_count": len(ids), "qualified_xy_triangle_area_sum_m2": float(cdf[-1]),
                           "group_count": groups, "failure_reason": None if complete else "max_attempts_exhausted",
                           "algorithm": "schema2-area-barycentric-mesh-column-v1",
                           "clearance_lower_offset_m": ground_band,
                           "multiple_ground_layer_tolerance_m": .15,
                           "grouping": "XY connected components at <= group_distance_m",
                           "pose_convention": "T_world_camera optical X-right Y-down Z-forward; xyzw w>=0; body Rz(yaw)Ry(-pitch)Rx(roll)",
                           "limitations": [
                               "N<=10000 is a request bound, not a guarantee that scene geometry or requested spacing can yield N samples.",
                               "Uses authored static triangle mesh only; NuRec appearance may differ.",
                               "Semantic candidates are model labels, not human ground truth or verified lanes.",
                               "Exact centre ground support; no interpolation or support guarantee over a vehicle footprint.",
                               "Conservative square-column triangle intersection rejects walls/overhangs from ground band to camera clearance.",
                               "Sub-ground-band clutter and unseen/unmeshed obstacles are not certified.",
                               "Multiple eligible ground layers at same XY >0.15m apart are rejected without floor selection.",
                               "No view-frustum visibility or image-quality guarantee; capture acceptance must check RGB separately.",
                               "XY triangle area sum may include overlapping reconstructed surfaces; it is not unique road area."]}}
