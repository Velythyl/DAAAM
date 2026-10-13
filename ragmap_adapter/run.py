"""Run DAAAM (ROS 2 path) on one RAGMAP trajectory, offline and headless.

    ragmap-run --input /input --output /output [key=value ...]            # the map
    ragmap-run summarize --input /input --output /output [key=value ...]  # region summaries

**What runs (the map).** Upstream's evaluated pipeline, the ROS 2 path of
``RUNNING.md`` (the ROS-free ``hydra_python`` path never receives embeddings,
so Hydra's embedding-based object merging silently switches off there):

1. The trajectory's frames are written to a rosbag2 (sqlite3) the way
   DAAAM-ROS's dataloader ``BagWriter`` writes one: per frame ``/tf`` first,
   then ``/clock``, then RGB (``bgr8``), depth (``16UC1``, millimetres) and
   ``CameraInfo``, on the topics and TF frames of the chosen launch profile.
   Frames are subsampled to 10 Hz by timestamp (DAAAM's frame-count settings
   -- ``query_interval_frames``, ``min_obs_per_track``, BotSort's
   ``track_buffer`` -- assume 10 Hz; a 30 fps render becomes every third
   frame).
2. Upstream's launch file for the profile (``launch/*_headless.launch.yaml``:
   upstream's, with only the visualizers dropped and the output directory an
   argument) starts the DAAAM node (FastSAM-x, BotSort with CLIP ReID, the
   CVXPY frame-selection MIP, DAM-3B region captions) and Hydra/Khronos. The
   bag is played with ``ros2 bag play --clock`` at ``play_rate`` (0.25; upstream
   plays in real time, approved deviation) once the node and its grounding worker report ready.
3. ``scripts/postprocess_scene_graph.py`` (sentence-t5-xl embeddings),
   then ``cluster_places.launch.yaml`` (``interactive:=false``) -> regions, with
   upstream's (CODa) cluster_places config; the ``hm3d`` profile replaces its
   room finder with the one of its own Hydra config (``clio_dataset_khronos.yaml``).
4. The objects are exported to RAGMAP's object contract.

``summarize`` runs ``scripts/summarize_regions.py`` on the map's
``clustered_dsg.json`` (see ``summarize.py``).

**What the adapter adds or changes**, each recorded in ``run.json``:

- *TensorRT engines.* The launch files name ``FastSAM-x-640x480.engine`` and
  ``clip_general.engine``, which are GPU- and TensorRT-specific. They are built
  on first use with upstream's own export scripts (``export_fastsam_trt.py``
  with its defaults, ``export_vanilla_clip_engine.py`` with its defaults) and
  cached per (GPU, compute capability, TensorRT version) under ``/cache``.
- *Drain* (``drain=true``). Upstream's Hydra node exits when the bag's
  ``/clock`` disappears, and the launch then stops the DAAAM node, which saves
  only what was already grounded. The adapter holds ``/clock`` at the bag's
  last time (``clock_keeper.py``) until DAAAM's queues are empty and no
  correction has arrived for ``drain_quiet_s``, then releases it, so the same
  shutdown follows with the last query interval grounded. ``drain=false`` is
  upstream's behaviour.
- *Frame times.* Synthetic timestamps (renders numbered from 0) are shifted by
  ``CLOCK_BASE_S`` so ROS time is never zero (zero means "no clock yet" under
  ``use_sim_time``); exported times are shifted back.
- *RGB-depth pairing.* Each bag frame is one ``frames.jsonl`` record, whose
  ``rgb`` and ``depth`` are named by that record (paired by frame id upstream of
  here); a missing or unreadable file fails the run. DAAAM-ROS's CODa
  dataloader pairs by position in two sorted file lists instead, where one
  missing depth PNG would silently shift every later frame; that loader is not
  used.
- *DDS isolation*: a random ``ROS_DOMAIN_ID`` and localhost-only discovery, so
  two runs on one host cannot hear each other.

**OUTPUT** (``ragmap.objects.base``): ``objects.jsonl`` (``pointclouds/*.ply``),
``regions.jsonl`` (cluster_places rooms, unsummarised), ``run.json``, and the
raw DAAAM run under ``daaam/`` (``dsg.json``, ``dsg_updated.json``,
``clustered_dsg.json``, ``corrections.yaml``, ``background_objects.yaml``,
``keyframe_annotations.yaml``, grounding images, logs) for DAAAM's own query
agent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DAAAM_ROOT = Path(os.environ.get("DAAAM_ROOT", Path(__file__).resolve().parents[1]))
ADAPTER_DIR = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("DAAAM_WEIGHTS", "/weights"))
CACHE = Path(os.environ.get("DAAAM_CACHE", "/cache"))
SCHEMA = "ragmap.daaam/v1"
#: Synthetic frame times are shifted by this many seconds (ROS time 0 is "no clock").
CLOCK_BASE_S = 1_000_000_000.0
FASTSAM_ENGINE = "FastSAM-x-640x480.engine"
REID_ENGINE = "clip_general.engine"


@dataclass(frozen=True)
class Profile:
    """One upstream launch file and the bag layout it expects."""

    name: str
    launch: str
    rgb_topic: str
    depth_topic: str
    info_topic: str
    camera_frame: str
    #: The launch's robot_frame; equal to camera_frame when the sensor is the body.
    body_frame: str
    world_frame: str = "world"
    #: DAAAM-ROS's CODa dataloader crop-resizes to 640x480; HM3D bags are full size.
    resize: tuple[int, int] | None = None
    #: The launch's depth_ub, for run.json only.
    depth_ub: float = 0.0
    #: Hydra config (in daaam_ros/config/hydra_config) whose online ``rooms.room_finder``
    #: replaces the one in cluster_places_config_coda.yaml; None = that file unchanged.
    #: Upstream releases only the CODa (outdoor) cluster_places config ("min_room_size: 30
    #: # 30 outdoors"); indoors, its room finder clusters a Replica room's ~30 places into
    #: nothing, so the indoor profile clusters with its own launch's room finder.
    cluster_room_finder_from: str | None = None


PROFILES = {
    # hm3d_daaam_hydra.launch.yaml: topics /forward/..., sensor = robot frame
    # "forward", 1080x720 renders (its max_mask_region_area is 1080*720).
    "hm3d": Profile("hm3d", "hm3d_daaam_hydra_headless.launch.yaml",
                    "/forward/color/image_raw", "/forward/depth/image_rect_raw", "/forward/camera_info",
                    camera_frame="forward", body_frame="forward", depth_ub=3.0,
                    cluster_room_finder_from="clio_dataset_khronos.yaml"),
    # coda_daaam_hydra.launch.yaml + dataloader_coda_with_depth.launch.yaml:
    # /cam0/..., cam0_optical_frame under os1 (static), world -> os1 (dynamic).
    "coda": Profile("coda", "coda_daaam_hydra_headless.launch.yaml",
                    "/cam0/rgb_image", "/cam0/depth_image", "/cam0/camera_info",
                    camera_frame="cam0_optical_frame", body_frame="os1", resize=(640, 480), depth_ub=20.0),
}


class HydraStartError(RuntimeError):
    """Hydra never built its pipeline after the clock started (a hang seen at random)."""


@dataclass
class Settings:
    profile: str = "hm3d"
    target_hz: float = 10.0
    #: ros2 bag play --rate. Upstream plays in real time on its hardware; on an L40S
    #: the node keeps ~3 of 10 frames at 1.0, so the default is slowed to keep them all
    #: (sim time is unchanged, so DAAAM sees the same 10 Hz stream).
    play_rate: float = 0.25
    drain: bool = True
    drain_quiet_s: float = 60.0
    drain_timeout_s: float = 3600.0
    startup_timeout_s: float = 2400.0
    shutdown_timeout_s: float = 900.0
    #: Seconds after the bag starts within which Hydra must report "[Hydra Frontend] started!".
    hydra_start_timeout_s: float = 240.0
    #: Whole-launch attempts when Hydra hangs at start (HydraStartError).
    launch_attempts: int = 2
    sentence_model: str = "sentence-transformers/sentence-t5-xl"
    keep_bag: bool = False
    #: Optional 4x4 row-major body_from_camera (CODa: os1 <- cam0_optical); identity if absent.
    body_from_camera: list[float] | None = None
    extra_launch_args: list[str] = field(default_factory=list)

    @classmethod
    def from_overrides(cls, overrides: Sequence[str], meta: dict[str, Any]) -> "Settings":
        settings = cls()
        hint = meta.get("daaam") or {}
        if isinstance(hint, dict):
            for key, value in hint.items():
                settings._set(key, value)
        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"Override {item!r} is not key=value.")
            settings._set(key.strip(), json.loads(value) if value[:1] in "[{" else value)
        if settings.profile not in PROFILES:
            raise ValueError(f"profile={settings.profile!r}; known: {sorted(PROFILES)}")
        return settings

    def _set(self, key: str, value: Any) -> None:
        if key == "launch":
            self.extra_launch_args.append(str(value))
            return
        if not hasattr(self, key):
            raise ValueError(f"Unknown setting {key!r}.")
        current = getattr(self, key)
        if isinstance(current, bool):
            value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, str):
            value = str(value)
        setattr(self, key, value)


@dataclass(frozen=True)
class Frame:
    frame_index: int
    rgb: Path
    depth: Path
    pose: np.ndarray
    timestamp: float


def _git_sha() -> str:
    sha = os.environ.get("DAAAM_GIT_SHA", "").strip()
    return sha if sha and sha != "unknown" else "unknown"


def log(message: str) -> None:
    print(f"[ragmap-daaam {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def read_input(input_dir: Path) -> tuple[dict[str, Any], list[Frame]]:
    meta = json.loads((input_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("up_axis") != "z":
        raise ValueError(f"meta.json up_axis is {meta.get('up_axis')!r}; DAAAM's places and regions need 'z'.")
    frames: list[Frame] = []
    for line_no, raw in enumerate((input_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if record.get("timestamp") is None:
            raise ValueError(f"frames.jsonl:{line_no} has no timestamp; DAAAM runs on a timed stream.")
        frames.append(Frame(
            frame_index=int(record["frame_index"]),
            rgb=input_dir / record["rgb"],
            depth=input_dir / record["depth"],
            pose=np.asarray(record["pose"], dtype=np.float64).reshape(4, 4),
            timestamp=float(record["timestamp"]),
        ))
    if not frames:
        raise ValueError(f"{input_dir / 'frames.jsonl'} holds no frames.")
    frames.sort(key=lambda frame: frame.timestamp)
    return meta, frames


def subsample(frames: Sequence[Frame], target_hz: float) -> list[Frame]:
    """Keep a frame when at least ~1/target_hz has passed since the last kept one."""

    period = 1.0 / target_hz
    kept: list[Frame] = []
    for frame in frames:
        if not kept or frame.timestamp - kept[-1].timestamp >= 0.9 * period:
            kept.append(frame)
    return kept


def gravity_check(frames: Sequence[Frame]) -> dict[str, Any]:
    """The world's up axis as the trajectory suggests it (minus the mean camera +y).

    Hydra's places and cluster_places' regions assume +z up; the contract says
    ``up_axis: z``. A level camera's +y (OpenCV: down) averages to -up, so this
    is recorded, and a world whose estimate is far from +z fails the run rather
    than producing nonsense places.
    """

    down = np.mean([frame.pose[:3, 1] for frame in frames], axis=0)
    up = -down / max(float(np.linalg.norm(down)), 1e-9)
    angle = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return {"estimated_up": up.tolist(), "angle_to_z_deg": angle, "consistency": float(np.linalg.norm(down))}


def clock_offset(frames: Sequence[Frame]) -> float:
    """Seconds added to every frame time in the bag (0 for real epoch times)."""

    return CLOCK_BASE_S if frames[0].timestamp < 1e8 else 0.0


# ---------------------------------------------------------------------------
# Bag
# ---------------------------------------------------------------------------


def _transform(parent: str, child: str, matrix: np.ndarray, stamp_ns: int) -> Any:
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation

    message = TransformStamped()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    message.header.frame_id = parent
    message.child_frame_id = child
    message.transform.translation.x = float(matrix[0, 3])
    message.transform.translation.y = float(matrix[1, 3])
    message.transform.translation.z = float(matrix[2, 3])
    qx, qy, qz, qw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    message.transform.rotation.x = float(qx)
    message.transform.rotation.y = float(qy)
    message.transform.rotation.z = float(qz)
    message.transform.rotation.w = float(qw)
    return message


def write_bag(
    bag_dir: Path, frames: Sequence[Frame], meta: dict[str, Any], profile: Profile,
    body_from_camera: np.ndarray, offset: float,
) -> dict[str, Any]:
    """The bag DAAAM-ROS's ``BagWriter`` would write for these frames."""

    import cv2
    from cv_bridge import CvBridge
    from rclpy.serialization import serialize_message
    from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo
    from tf2_msgs.msg import TFMessage

    if bag_dir.exists():
        shutil.rmtree(bag_dir)
    bag_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    topics = [
        ("/tf_static", "tf2_msgs/msg/TFMessage"), ("/tf", "tf2_msgs/msg/TFMessage"),
        ("/clock", "rosgraph_msgs/msg/Clock"), (profile.rgb_topic, "sensor_msgs/msg/Image"),
        (profile.depth_topic, "sensor_msgs/msg/Image"), (profile.info_topic, "sensor_msgs/msg/CameraInfo"),
    ]
    for topic_id, (name, kind) in enumerate(topics):
        writer.create_topic(TopicMetadata(id=topic_id, name=name, type=kind, serialization_format="cdr"))
    bridge = CvBridge()
    fx, fy, cx, cy = (float(meta[key]) for key in ("fx", "fy", "cx", "cy"))
    width, height = int(meta["width"]), int(meta["height"])
    depth_scale = float(meta.get("depth_scale", 1000.0))
    resize_info: dict[str, Any] | None = None
    camera_from_body = np.linalg.inv(body_from_camera)
    static_written = False
    for frame in frames:
        stamp_ns = int(round((frame.timestamp + offset) * 1e9))
        bgr = cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth_raw is None:
            raise FileNotFoundError(f"Frame {frame.frame_index}: cannot read {frame.rgb} or {frame.depth}.")
        depth_m = depth_raw.astype(np.float32) / depth_scale
        if depth_m.shape != bgr.shape[:2]:
            raise ValueError(f"Frame {frame.frame_index}: depth {depth_m.shape} is not RGB {bgr.shape[:2]}.")
        k = (fx, fy, cx, cy)
        out_w, out_h = width, height
        if profile.resize is not None:
            # DAAAM-ROS's dataloader (resize_images, crop_resize, 640x480).
            from daaam_ros.nodes.dataloader.image_utils import (
                resize_depth_image, resize_rgb_image, update_intrinsics_for_resize,
            )
            from daaam_ros.nodes.dataloader.models import CameraIntrinsics

            out_w, out_h = profile.resize
            bgr, crop = resize_rgb_image(bgr, out_w, out_h)
            depth_m, _ = resize_depth_image(depth_m, out_w, out_h)
            intrinsics = update_intrinsics_for_resize(
                CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height),
                (width, height), crop, (out_w, out_h),
            )
            k = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
            resize_info = {"from": [width, height], "to": [out_w, out_h], "crop": list(crop), "intrinsics": list(k)}
        header_stamp = (stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000)
        # BagWriter: static TF once, then per frame /tf, /clock, images, info.
        if not static_written and profile.body_frame != profile.camera_frame:
            static = TFMessage()
            static.transforms.append(_transform(profile.body_frame, profile.camera_frame, body_from_camera, stamp_ns))
            writer.write("/tf_static", serialize_message(static), stamp_ns)
            static_written = True
        world_from_body = frame.pose @ camera_from_body
        tf = TFMessage()
        tf.transforms.append(_transform(profile.world_frame, profile.body_frame, world_from_body, stamp_ns))
        writer.write("/tf", serialize_message(tf), stamp_ns)
        clock = Clock()
        clock.clock.sec, clock.clock.nanosec = header_stamp
        writer.write("/clock", serialize_message(clock), stamp_ns)
        rgb_msg = bridge.cv2_to_imgmsg(np.ascontiguousarray(bgr), encoding="bgr8")
        depth_mm = (depth_m * 1000).astype(np.uint16)
        depth_msg = bridge.cv2_to_imgmsg(np.ascontiguousarray(depth_mm), encoding="16UC1")
        info = CameraInfo()
        for message in (rgb_msg, depth_msg, info):
            message.header.stamp.sec, message.header.stamp.nanosec = header_stamp
            message.header.frame_id = profile.camera_frame
        info.width, info.height = out_w, out_h
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [k[0], 0.0, k[2], 0.0, k[1], k[3], 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [k[0], 0.0, k[2], 0.0, 0.0, k[1], k[3], 0.0, 0.0, 0.0, 1.0, 0.0]
        writer.write(profile.rgb_topic, serialize_message(rgb_msg), stamp_ns)
        writer.write(profile.depth_topic, serialize_message(depth_msg), stamp_ns)
        writer.write(profile.info_topic, serialize_message(info), stamp_ns)
    del writer  # closes the bag
    return {
        "frames": len(frames),
        "first_stamp_s": frames[0].timestamp + offset,
        "last_stamp_s": frames[-1].timestamp + offset,
        "resize": resize_info,
        "size_bytes": sum(path.stat().st_size for path in bag_dir.rglob("*") if path.is_file()),
    }


# ---------------------------------------------------------------------------
# TensorRT engines
# ---------------------------------------------------------------------------


def _engine_key() -> str:
    import tensorrt
    import torch

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return f"{safe}_sm{major}{minor}_trt{tensorrt.__version__}"


def prepare_checkpoints(env: dict[str, str]) -> dict[str, Any]:
    """Put FastSAM-x and the ReID engine where the launch files look for them.

    Engines are built with upstream's own export scripts on a cache miss.
    """

    import fcntl

    fastsam_pt = WEIGHTS / "fastsam" / "FastSAM-x.pt"
    clip_pt = WEIGHTS / "clip" / "ViT-B-16.pt"
    for path in (fastsam_pt, clip_pt):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing (mount the DAAAM weights at {WEIGHTS}).")
    key = _engine_key()
    cache_dir = CACHE / "trt" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = DAAAM_ROOT / "checkpoints"
    (checkpoints / "fastsam").mkdir(parents=True, exist_ok=True)
    (checkpoints / "reid_weights").mkdir(parents=True, exist_ok=True)
    built: list[str] = []
    timings: dict[str, float] = {}
    with open(cache_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache_dir / FASTSAM_ENGINE).is_file():
            started = time.time()
            shutil.copy2(fastsam_pt, checkpoints / "fastsam" / "FastSAM-x.pt")
            subprocess.run([sys.executable, "scripts/export_fastsam_trt.py", "--model_name", "FastSAM-x"],
                           cwd=DAAAM_ROOT, env=env, check=True)
            shutil.copy2(checkpoints / "fastsam" / FASTSAM_ENGINE, cache_dir / (FASTSAM_ENGINE + ".tmp"))
            os.replace(cache_dir / (FASTSAM_ENGINE + ".tmp"), cache_dir / FASTSAM_ENGINE)
            built.append(FASTSAM_ENGINE)
            timings["fastsam_engine_build_seconds"] = time.time() - started
        if not (cache_dir / REID_ENGINE).is_file():
            started = time.time()
            clip_cache = Path(env.get("HOME", "/root")) / ".cache" / "clip"
            clip_cache.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip_pt, clip_cache / "ViT-B-16.pt")
            subprocess.run([sys.executable, "scripts/export_vanilla_clip_engine.py"],
                           cwd=DAAAM_ROOT, env=env, check=True)
            shutil.copy2(checkpoints / "reid_weights" / REID_ENGINE, cache_dir / (REID_ENGINE + ".tmp"))
            os.replace(cache_dir / (REID_ENGINE + ".tmp"), cache_dir / REID_ENGINE)
            built.append(REID_ENGINE)
            timings["reid_engine_build_seconds"] = time.time() - started
    for name, sub in ((FASTSAM_ENGINE, "fastsam"), (REID_ENGINE, "reid_weights")):
        target = checkpoints / sub / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(cache_dir / name)
    return {"key": key, "cache_dir": str(cache_dir), "built": built, **timings}


# ---------------------------------------------------------------------------
# The ROS run
# ---------------------------------------------------------------------------


def ros_env(work: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(random.SystemRandom().randint(1, 100))
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    # Never let ultralytics pip-install anything at run time (INSTALL.md: a bare
    # `tensorrt` would pull CUDA 13 libraries); a missing requirement fails loudly.
    env["YOLO_AUTOINSTALL"] = "False"
    env.setdefault("HOME", "/root")
    overrides = work / "tf_overrides.yaml"
    overrides.write_text("/tf_static: {depth: 1, durability: transient_local}\n", encoding="utf-8")
    env["ROS_STATIC_TRANSFORM_BROADCASTER_QOS_OVERRIDES"] = str(overrides)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(DAAAM_ROOT), env.get("PYTHONPATH", "")]))
    return env


def _spawn(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log(" ".join(command))
    handle = open(log_path, "ab")
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                            cwd=DAAAM_ROOT, start_new_session=True)


def _stop(process: subprocess.Popen, timeout: float) -> int:
    if process.poll() is not None:
        return process.returncode
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 30.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            return process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    return process.wait()


def _grep_logs(root: Path, needle: str) -> bool:
    for path in root.rglob("*.log"):
        try:
            if needle in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _read_status(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def pipeline_config(work: Path) -> Path:
    """Upstream's ``config/pipeline_config.yaml`` with DAM-3B named by its local snapshot.

    DAM's loader (``dam.model.utils``, ``language_model.builder``) asks the Hub
    whether ``nvidia/DAM-3B`` exists before reading the cache, which fails
    offline; given a directory it reads it directly. Same weights (the cached
    snapshot), every other value unchanged; the DAAAM node still applies the
    launch file's parameters on top, as upstream.
    """

    import yaml
    from huggingface_hub import snapshot_download

    config = yaml.safe_load((DAAAM_ROOT / "config" / "pipeline_config.yaml").read_text(encoding="utf-8"))
    dam = config["workers"]["dam_grounding_config"]
    dam["dam_model_path"] = snapshot_download(dam["dam_model_path"], local_files_only=True)
    path = work / "pipeline_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_daaam(
    settings: Settings, profile: Profile, bag: Path, bag_info: dict[str, Any],
    work: Path, run_dir: Path, env: dict[str, str], timings: dict[str, float],
) -> dict[str, Any]:
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = work / "daaam_status.json"
    hold_file, stop_file = work / "clock.hold", work / "clock.stop"
    for path in (status_path, hold_file, stop_file):
        path.unlink(missing_ok=True)
    launch_env = dict(env)
    launch_env["RAGMAP_DAAAM_STATUS"] = str(status_path)
    launch_env["PYTHONPATH"] = os.pathsep.join([str(ADAPTER_DIR / "hook"), launch_env["PYTHONPATH"]])
    node_logs = DAAAM_ROOT / "output" / "logs"
    before = set(node_logs.glob("*")) if node_logs.exists() else set()
    run_dir.mkdir(parents=True, exist_ok=True)
    launch = _spawn([
        "ros2", "launch", str(ADAPTER_DIR / "launch" / profile.launch),
        f"output_dir:={run_dir}", f"log_path:={work / 'hydra'}",
        f"pipeline_config:={pipeline_config(work)}", *settings.extra_launch_args,
    ], logs / "launch.log", launch_env)
    keeper = None
    if settings.drain:
        last_ns = int(round(bag_info["last_stamp_s"] * 1e9)) + 1_000_000_000
        keeper = _spawn([sys.executable, "-m", "ragmap_adapter.clock_keeper", "--time-ns", str(last_ns),
                         "--hold-file", str(hold_file), "--stop-file", str(stop_file)],
                        logs / "clock_keeper.log", env)
    result: dict[str, Any] = {"drain": settings.drain, "play_rate": settings.play_rate}
    try:
        # Ready: the node built its pipeline (the status hook is running) and
        # the grounding worker finished loading DAM-3B and CLIP.
        started = time.time()
        while True:
            if launch.poll() is not None:
                raise RuntimeError(f"The DAAAM launch exited ({launch.returncode}) before it was ready; see {logs / 'launch.log'}.")
            new_logs = [path for path in node_logs.glob("*") if path not in before] if node_logs.exists() else []
            ready_node = status_path.exists()
            ready_worker = any(_grep_logs(path, "Finished initializing successfully") for path in new_logs)
            if any(_grep_logs(path, "Failed to initialize DAM agent") for path in new_logs):
                raise RuntimeError("DAAAM's grounding worker could not load DAM-3B (its log says so).")
            if ready_node and ready_worker:
                break
            if time.time() - started > settings.startup_timeout_s:
                raise TimeoutError(f"DAAAM was not ready after {settings.startup_timeout_s:.0f} s.")
            time.sleep(5)
        timings["startup_seconds"] = time.time() - started
        time.sleep(5.0)
        log("DAAAM ready; playing the bag.")
        started = time.time()
        player = subprocess.Popen(
            ["ros2", "bag", "play", str(bag), "--clock", "--rate", str(settings.play_rate),
             "--qos-profile-overrides-path", env["ROS_STATIC_TRANSFORM_BROADCASTER_QOS_OVERRIDES"]],
            env=env, cwd=DAAAM_ROOT, stdout=open(logs / "bag_play.log", "ab"), stderr=subprocess.STDOUT,
        )
        # Hydra builds its pipeline only once the clock runs. It has been seen to hang
        # before "[Hydra Frontend] started!" for the whole run (office3, 1 of 9 runs),
        # leaving no dsg.json; fail fast so main() can relaunch instead.
        hydra_seen = False
        while player.poll() is None:
            if not hydra_seen:
                hydra_seen = "[Hydra Frontend] started!" in (logs / "launch.log").read_text(errors="replace")
                if not hydra_seen and time.time() - started > settings.hydra_start_timeout_s:
                    _stop(player, 10.0)
                    raise HydraStartError(f"Hydra did not start within {settings.hydra_start_timeout_s:.0f} s of the bag.")
            time.sleep(2.0)
        timings["play_seconds"] = time.time() - started
        if player.returncode:
            raise RuntimeError(f"ros2 bag play exited {player.returncode}; see {logs / 'bag_play.log'}.")
        after_play = _read_status(status_path)
        result["frames_received_at_bag_end"] = after_play.get("frame_count")
        if keeper is not None:
            hold_file.touch()
            started = time.time()
            quiet_since, last_seen = time.time(), None
            while True:
                status = _read_status(status_path)
                queues = status.get("queues") or {}
                busy = any(size > 0 for size in queues.values()) or status.get("pending_track_ids", 0) > 0
                marker = (status.get("frame_count"), status.get("corrections_stored"), tuple(sorted(queues.items())))
                if marker != last_seen or busy:
                    last_seen, quiet_since = marker, time.time()
                if time.time() - quiet_since >= settings.drain_quiet_s:
                    result["drain_status"] = "drained"
                    break
                if time.time() - started > settings.drain_timeout_s:
                    result["drain_status"] = "timeout"
                    break
                if launch.poll() is not None:
                    raise RuntimeError(f"The DAAAM launch exited ({launch.returncode}) while draining; see {logs / 'launch.log'}.")
                time.sleep(2.0)
            timings["drain_seconds"] = time.time() - started
            result["status_after_drain"] = _read_status(status_path)
            stop_file.touch()
        # Hydra exits once no /clock is left; the launch then shuts down and
        # the DAAAM node saves its run (upstream's own shutdown path).
        started = time.time()
        try:
            launch.wait(timeout=settings.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            log("The launch did not exit after the clock stopped; stopping it.")
            result["forced_shutdown"] = True
            _stop(launch, 120.0)
        timings["shutdown_seconds"] = time.time() - started
        result["launch_returncode"] = launch.returncode
    finally:
        stop_file.touch()
        if keeper is not None:
            _stop(keeper, 10.0)
        _stop(launch, 120.0)
        new_logs = [path for path in node_logs.glob("*") if path not in before] if node_logs.exists() else []
        for path in new_logs:
            shutil.copytree(path, logs / "daaam_node" / path.name, dirs_exist_ok=True)
    return result


def find_run_output(run_dir: Path) -> Path:
    candidates = sorted(path for path in run_dir.glob("out_*") if (path / "dsg.json").is_file())
    if not candidates:
        raise RuntimeError(f"DAAAM saved no dsg.json under {run_dir}; see the launch log.")
    return candidates[-1]


def postprocess(out: Path, settings: Settings, env: dict[str, str], logs: Path) -> None:
    with open(logs / "postprocess.log", "ab") as handle:
        subprocess.run([sys.executable, "scripts/postprocess_scene_graph.py", "--data-dir", str(out),
                        "--sentence-model-name", settings.sentence_model],
                       cwd=DAAAM_ROOT, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)
    if not (out / "dsg_updated.json").is_file():
        raise RuntimeError(f"postprocess_scene_graph.py wrote no {out / 'dsg_updated.json'}.")


def cluster_places_config(profile: Profile, work: Path, env: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    """The cluster_places config for ``profile``: upstream's CODa file, with the room
    finder of the profile's Hydra config when ``cluster_room_finder_from`` is set."""
    import yaml

    prefix = subprocess.run(["ros2", "pkg", "prefix", "daaam_ros"], env=env, check=True,
                            capture_output=True, text=True).stdout.strip()
    share = Path(prefix) / "share" / "daaam_ros" / "config"
    source = share / "cluster_places_config_coda.yaml"
    if profile.cluster_room_finder_from is None:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        return source, config["room_finder"]
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    hydra = yaml.safe_load((share / "hydra_config" / profile.cluster_room_finder_from).read_text(encoding="utf-8"))
    config["room_finder"] = dict(hydra["backend"]["update_functors"]["rooms"]["room_finder"])
    path = work / f"cluster_places_config_{profile.name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, config["room_finder"]


def cluster_places(out: Path, env: dict[str, str], logs: Path, timeout: float, config: Path) -> None:
    process = _spawn(["ros2", "launch", "daaam_ros", "cluster_places.launch.yaml",
                      f"data_dir:={out}", "interactive:=false", f"cluster_places_config_path:={config}"],
                     logs / "cluster_places.log", env)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop(process, 30.0)
        raise TimeoutError(f"cluster_places did not finish in {timeout:.0f} s.")
    finally:
        _stop(process, 30.0)
    if not (out / "clustered_dsg.json").is_file():
        raise RuntimeError(f"cluster_places wrote no {out / 'clustered_dsg.json'}; see {logs / 'cluster_places.log'}.")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["summarize"]:
        from ragmap_adapter.summarize import main as summarize_main

        return summarize_main(argv[1:])
    if argv[:1] == ["navqa"]:
        from ragmap_adapter.navqa import main as navqa_main

        return navqa_main(argv[1:])
    parser = argparse.ArgumentParser(prog="ragmap-run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value settings (see Settings)")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {"schema": SCHEMA, "status": "running", "daaam_git_sha": _git_sha(),
                           "overrides": list(args.overrides)}
    timings: dict[str, float] = {}
    started = time.time()
    work = output / "_work"
    try:
        meta, frames = read_input(args.input)
        settings = Settings.from_overrides(args.overrides, meta)
        profile = PROFILES[settings.profile]
        run["settings"] = {key: getattr(settings, key) for key in settings.__dataclass_fields__}
        selected = subsample(frames, settings.target_hz)
        gravity = gravity_check(selected)
        run["gravity"] = gravity
        if gravity["angle_to_z_deg"] > 30.0:
            raise ValueError(f"The trajectory's up axis is {gravity['angle_to_z_deg']:.0f} deg from +z; "
                             "Hydra's places need a z-up world (meta.json says up_axis z).")
        offset = clock_offset(selected)
        body_from_camera = (np.asarray(settings.body_from_camera, dtype=np.float64).reshape(4, 4)
                            if settings.body_from_camera is not None else np.eye(4))
        work.mkdir(parents=True, exist_ok=True)
        env = ros_env(work)
        t0 = time.time()
        run["engines"] = prepare_checkpoints(env)
        timings["engines_seconds"] = time.time() - t0
        t0 = time.time()
        bag = work / "bag"
        bag_info = write_bag(bag, selected, meta, profile, body_from_camera, offset)
        timings["bag_seconds"] = time.time() - t0
        run["bag"] = {**bag_info, "clock_offset_s": offset, "frames_input": len(frames),
                      "target_hz": settings.target_hz, "profile": profile.name}
        run_dir = output / "daaam_runs"
        attempts = []
        for attempt in range(1, settings.launch_attempts + 1):
            try:
                run["ros"] = run_daaam(settings, profile, bag, bag_info, work, run_dir, env, timings)
                break
            except HydraStartError as exc:
                attempts.append(str(exc))
                log(f"attempt {attempt}: {exc}")
                if attempt == settings.launch_attempts:
                    raise
                shutil.rmtree(run_dir, ignore_errors=True)
                shutil.move(str(work / "logs"), str(work / f"logs_failed_attempt{attempt}"))
        run["launch_failed_attempts"] = attempts
        out = find_run_output(run_dir)
        t0 = time.time()
        postprocess(out, settings, env, work / "logs")
        timings["postprocess_seconds"] = time.time() - t0
        t0 = time.time()
        cluster_config, room_finder = cluster_places_config(profile, work, env)
        run["cluster_places"] = {"config": str(cluster_config), "room_finder": room_finder}
        cluster_places(out, env, work / "logs", timeout=1800.0, config=cluster_config)
        shutil.copy2(cluster_config, out / "cluster_places_config.yaml")
        timings["cluster_places_seconds"] = time.time() - t0
        final = output / "daaam"
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(out), final)
        shutil.copytree(work / "logs", final / "ragmap_logs", dirs_exist_ok=True)
        from ragmap_adapter.export import export

        counts = export(final, selected, offset, output)
        run["counts"] = {**counts, "frames_input": len(frames), "frames_bag": len(selected),
                         "frames_received": (run["ros"].get("status_after_drain") or {}).get("frame_count",
                                              run["ros"].get("frames_received_at_bag_end"))}
        if not settings.keep_bag:
            shutil.rmtree(bag, ignore_errors=True)
        run["status"] = "ok"
        return 0
    except BaseException as exc:  # noqa: BLE001 - recorded, then the exit code
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["traceback"] = traceback.format_exc()
        traceback.print_exc()
        return 1
    finally:
        timings["total_seconds"] = time.time() - started
        run["timings"] = timings
        (output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
