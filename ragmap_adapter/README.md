# RAGMAP adapter

`ghcr.io/velythyl/daaam` runs DAAAM's evaluated pipeline -- the ROS 2 path
(DAAAM node + Hydra/Khronos from `DAAAM-ROS`), `postprocess_scene_graph.py`,
`cluster_places`, and `summarize_regions.py` -- behind RAGMAP's object-mapping
and area-mapping container contracts (`ragmap/objects/base.py`,
`ragmap/areas/base.py` in RAGMAP). Upstream code is untouched; everything added
is in `ragmap_adapter/`, `Dockerfile.ragmap` and `.github/workflows/ghcr.yml`.
`run.py`'s and `summarize.py`'s docstrings list exactly what the adapter
supplies and where it departs from upstream.

    # the map (one GPU)
    podman run --rm --device nvidia.com/gpu=all \
      -v <input>:/input:ro -v <output>:/output \
      -v <daaam weights>:/weights:ro -v <HF hub cache>:/hf:ro -v <engine cache>:/cache \
      -e HF_HUB_CACHE=/hf ghcr.io/velythyl/daaam:<tag> --input /input --output /output [key=value ...]

    # region summaries (no GPU; the LLM is an OpenAI-compatible chat endpoint)
    podman run --rm --network host -v <map output>:/input:ro -v <output>:/output \
      -e RAGMAP_VLM_BASE_URL=http://127.0.0.1:PORT -e RAGMAP_VLM_MODEL=<model> \
      ghcr.io/velythyl/daaam:<tag> summarize --input /input --output /output

Map settings (`key=value`, see `run.py` `Settings`): `profile=hm3d|coda`
(upstream's launch file: HM3D indoor, CODa outdoor), `target_hz=10`,
`play_rate=1.0`, `drain=true`, `drain_quiet_s=60`, `body_from_camera=[16 floats]`
(CODa: os1 <- cam0_optical), `launch=<name>:=<value>` (extra launch arguments).

## Mounts

- `/weights` (read-only): `fastsam/FastSAM-x.pt` (ultralytics v8.2.0 asset) and
  `clip/ViT-B-16.pt` (OpenAI CLIP, for the BotSort ReID engine).
- `/hf` (read-only, `HF_HUB_CACHE`): `nvidia/DAM-3B`,
  `sentence-transformers/sentence-t5-xl`, `timm/vit_large_patch14_clip_224.openai`
  (OpenCLIP ViT-L-14 openai, the HM3D select-frame CLIP),
  `facebook/PE-Core-L14-336` (the CODa one).
- `/cache` (read-write): `trt/<GPU>_sm<cc>_trt<version>/{FastSAM-x-640x480.engine,
  clip_general.engine}`, built on first use by upstream's export scripts.

## Input and output

INPUT `/input`: `meta.json` (`fx fy cx cy width height depth_scale up_axis=z`,
optional `daaam: {settings}`), `frames.jsonl` (`frame_index observation_id rgb
depth pose timestamp`; pose is a row-major 4x4 OpenCV camera-to-world in
metres), `rgb/`, `depth/` (uint16 PNG).

OUTPUT `/output`: `objects.jsonl` (`id label caption centroid bbox_min bbox_max
pointcloud frame_indices extra`; `extra` has `sentence_t5_xl`, `clip_feature`,
`first_seen_s`, `last_seen_s`, `layer`), `pointclouds/*.ply`, `regions.jsonl`,
`run.json`, and the raw run in `daaam/`.
