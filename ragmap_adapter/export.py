"""DAAAM's finished scene graph -> RAGMAP's object contract (and its regions).

Read, never re-derived:

- ``dsg_updated.json`` (``postprocess_scene_graph.py``'s output): the OBJECTS
  layer (Hydra/Khronos objects, DAAAM's semantic label on each) and the
  BACKGROUND_OBJECTS layer (DAAAM's objects that Hydra filtered, position
  only), each with ``sentence_embedding_feature`` (sentence-t5-xl of the DAM
  description) in its metadata; the graph metadata's ``features`` also hold
  the select-frame CLIP feature per semantic label.
- ``corrections.yaml``: ``label_names`` -- the DAM description of every
  semantic label and its temporal history (first/last observation times).
- ``clustered_dsg.json`` (``cluster_places``): the ROOMS layer, and
  ``daaam.utils.scene_graph_utils.objects_by_region`` (upstream's own
  object-to-region assignment, the one ``summarize_regions.py`` uses).

**Points.** A Khronos object's mesh (``KhronosObjectAttributes.mesh()``) when
it has one, else the scene mesh vertices it is connected to
(``mesh_connections``), else its bounding-box corners; a background object is
its one position. ``extra.points_source`` says which.

Ids: ``daaam_O<n>`` (OBJECTS) and ``daaam_o<n>`` (BACKGROUND_OBJECTS), from
the node symbol's category id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


def _vertices(mesh: Any) -> np.ndarray:
    if mesh is None:
        return np.zeros((0, 3))
    try:
        vertices = np.asarray(mesh.get_vertices(), dtype=np.float64)
    except Exception:  # noqa: BLE001 - an attribute without a mesh
        return np.zeros((0, 3))
    if vertices.size == 0:
        return np.zeros((0, 3))
    if vertices.ndim == 2 and vertices.shape[0] in (3, 6) and vertices.shape[1] not in (3, 6):
        vertices = vertices.T
    return vertices[:, :3]


def _box(attrs: Any) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    box = getattr(attrs, "bounding_box", None)
    if box is None or not box.is_valid():
        return None, None, np.zeros((0, 3))
    corners = np.asarray(box.corners(), dtype=np.float64).reshape(-1, 3)
    return corners.min(axis=0), corners.max(axis=0), corners


def object_points(graph: Any, attrs: Any) -> tuple[np.ndarray, str]:
    """Points of one OBJECTS node, in the graph's world frame, and their source."""

    lo, hi, corners = _box(attrs)
    mesh = getattr(attrs, "mesh", None)
    points = _vertices(mesh() if callable(mesh) else None)
    if len(points):
        if lo is not None:
            pad = 0.05 + 0.05 * float(np.max(hi - lo))
            inside = np.all((points >= lo - pad) & (points <= hi + pad), axis=1).mean()
            if inside < 0.5:
                # Khronos stores some meshes relative to the box centre.
                shifted = points + np.asarray(attrs.bounding_box.world_P_center, dtype=np.float64)
                if np.all((shifted >= lo - pad) & (shifted <= hi + pad), axis=1).mean() > inside:
                    return shifted, "khronos_mesh_box_frame"
        return points, "khronos_mesh"
    connections = list(getattr(attrs, "mesh_connections", []) or [])
    if connections and graph.has_mesh():
        scene = _vertices(graph.mesh)
        index = np.asarray([i for i in connections if 0 <= i < len(scene)], dtype=np.int64)
        if len(index):
            return scene[index], "scene_mesh_connections"
    if len(corners):
        return corners, "bounding_box_corners"
    return np.asarray(attrs.position, dtype=np.float64).reshape(1, 3), "position"


def _write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        handle.write(header.encode("ascii"))
        handle.write(np.asarray(points, dtype="<f4").tobytes())


def _label_names(corrections: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(entry["label"]): entry for entry in corrections.get("label_names") or [] if "label" in entry}


def _frames_of(times: Sequence[float], frame_times: np.ndarray, frame_indices: Sequence[int]) -> list[int]:
    if not len(times) or not len(frame_times):
        return []
    positions = np.clip(np.searchsorted(frame_times, np.asarray(times, dtype=np.float64)), 0, len(frame_times) - 1)
    out = set()
    for time_s, position in zip(times, positions):
        best = min((position, max(position - 1, 0)), key=lambda i: abs(frame_times[i] - time_s))
        out.add(int(frame_indices[best]))
    return sorted(out)


def _floats(values: Any) -> list[float] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array.tolist() if array.size else None


def export(daaam_dir: Path, frames: Sequence[Any], offset: float, output: Path) -> dict[str, Any]:
    import spark_dsg

    from daaam.utils.scene_graph_utils import objects_by_region

    graph = spark_dsg.DynamicSceneGraph.load(str(daaam_dir / "dsg_updated.json"))
    clustered = spark_dsg.DynamicSceneGraph.load(str(daaam_dir / "clustered_dsg.json"))
    corrections = yaml.safe_load((daaam_dir / "corrections.yaml").read_text(encoding="utf-8")) or {}
    names = _label_names(corrections)
    features = dict(graph.metadata.get()).get("features", {}) or {}
    frame_times = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    frame_indices = [frame.frame_index for frame in frames]

    rows: list[dict[str, Any]] = []
    counts = {"objects": 0, "background_objects": 0, "without_description": 0, "without_sentence_feature": 0}
    sources: dict[str, int] = {}

    def add(node: Any, layer: str, points: np.ndarray, points_source: str) -> None:
        attrs = node.attributes
        symbol = node.id
        object_id = f"daaam_{symbol.category}{symbol.category_id}"
        semantic = int(attrs.semantic_label)
        entry = names.get(semantic, {})
        meta = dict(attrs.metadata.get())
        description = str(entry.get("name") or meta.get("description") or "").strip()
        if not description or description == "unknown":
            counts["without_description"] += 1
        history = entry.get("temporal_history") or {}
        times = [float(t) - offset for t in history.get("timestamps") or []]
        sentence = meta.get("sentence_embedding_feature") or (features.get(str(semantic)) or {}).get("sentence_embedding_feature")
        if not sentence:
            counts["without_sentence_feature"] += 1
        clip = (features.get(str(semantic)) or {}).get("clip_feature")
        cloud = f"pointclouds/{object_id}.ply"
        _write_ply(output / cloud, points)
        sources[points_source] = sources.get(points_source, 0) + 1
        first = history.get("first_observed")
        last = history.get("last_observed")
        rows.append({
            "id": object_id,
            "label": description or "unknown",
            "caption": description or None,
            "crop": None,
            "centroid": np.asarray(attrs.position, dtype=np.float64).reshape(3).tolist(),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
            "pointcloud": cloud,
            "frame_indices": _frames_of(times, frame_times, frame_indices),
            "floor": None,
            "room": None,
            "extra": {
                "layer": layer,
                "node": str(symbol),
                "semantic_label": semantic,
                "points_source": points_source,
                "n_points": int(len(points)),
                "first_seen_s": float(first) - offset if first is not None else None,
                "last_seen_s": float(last) - offset if last is not None else None,
                "observation_count": history.get("observation_count") or meta.get("observations"),
                # sentence-t5-xl of the DAM description (postprocess_scene_graph.py), L2-normalised.
                "sentence_t5_xl": _floats(sentence),
                # DAAAM's select-frame CLIP feature of the object (selectframe_clip_* settings).
                "clip_feature": _floats(clip),
                "in_hydra": meta.get("in_hydra"),
            },
        })

    for node in graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        points, source = object_points(graph, node.attributes)
        add(node, "OBJECTS", points, source)
        counts["objects"] += 1
    if graph.has_layer("BACKGROUND_OBJECTS"):
        for node in graph.get_layer("BACKGROUND_OBJECTS").nodes:
            add(node, "BACKGROUND_OBJECTS", np.asarray(node.attributes.position, dtype=np.float64).reshape(1, 3), "position")
            counts["background_objects"] += 1
    with (output / "objects.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # Regions: the clustered graph's ROOMS, with upstream's object assignment.
    regions: list[dict[str, Any]] = []
    if clustered.has_layer(spark_dsg.DsgLayers.ROOMS):
        objs, _, bg_objs = objects_by_region(clustered)
        rooms = {node.id.value: node for node in clustered.get_layer(spark_dsg.DsgLayers.ROOMS).nodes}
        by_category = {node.id.category_id: node for node in rooms.values()}
        for region_id in sorted(set(objs) | set(bg_objs) | set(rooms)):
            node = rooms.get(region_id) or by_category.get(region_id)
            if node is None:
                continue
            children = [f"daaam_O{i}" for i in objs.get(region_id, [])] + [f"daaam_o{i}" for i in bg_objs.get(region_id, [])]
            regions.append({
                "id": f"daaam_{node.id.category}{node.id.category_id}",
                "node": str(node.id),
                "region_key": int(region_id),
                "centroid": np.asarray(node.attributes.position, dtype=np.float64).reshape(3).tolist(),
                "children": children,
                "num_places": len(list(node.children())),
            })
        regions = list({region["id"]: region for region in regions}.values())
    with (output / "regions.jsonl").open("w", encoding="utf-8") as handle:
        for region in regions:
            handle.write(json.dumps(region) + "\n")
    return {**counts, "points_source": sources, "regions": len(regions),
            "objects_in_regions": sum(len(region["children"]) for region in regions)}
