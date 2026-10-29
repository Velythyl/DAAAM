"""DAM-3B vs a served VLM on the exact inputs DAM saw in a finished DAAAM build (ours, 2026-10-29).

    python compare_describers.py <grounding worker dir> <vllm url> <model> <out dir> [n_objects=20]

The worker dir is ``.../daaam/worker_GroundingWorker-0`` of a build: ``grounding_annotations/*.yaml``
(semantic_id -> DAM description), ``grounding_images_plain/*.jpg`` (the frame) and
``grounding_images_masks_only/*.png`` (16-bit, pixel value = semantic_id). Objects are taken spread over
the batches, larger than 1500 px, plus two whole-image entries. Writes ``comparison.json`` and, per item,
the two images the VLM saw (``<k>_full.jpg``, ``<k>_crop.jpg``).
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragmap_adapter.qwen_describer import describe, region_images  # noqa: E402


def main() -> None:
    worker, url, model, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((worker / "grounding_annotations").glob("*.yaml"))
    picks, wholes = [], []
    for path in files[:: max(1, len(files) // n)]:
        data = yaml.safe_load(path.read_text())
        entries = data.get("annotations") or data.get("objects") or data if isinstance(data, list) else \
            next(v for v in data.values() if isinstance(v, list))
        masks = np.asarray(Image.open(worker / "grounding_images_masks_only" / f"{path.stem}.png"))
        for e in entries:
            sid = int(e["semantic_id"])
            if e.get("is_full_image") or sid == -1:
                if len(wholes) < 2:
                    wholes.append((path, e))
                continue
            if (masks == sid).sum() > 1500:
                picks.append((path, e))
                break
        if len(picks) >= n:
            break
    results = []
    for k, (path, e) in enumerate(picks + wholes):
        frame = np.asarray(Image.open(worker / "grounding_images_plain" / f"{path.stem}.jpg").convert("RGB"))
        sid = int(e["semantic_id"])
        whole = e.get("is_full_image") or sid == -1
        mask = None
        if not whole:
            masks = np.asarray(Image.open(worker / "grounding_images_masks_only" / f"{path.stem}.png"))
            if masks.shape != frame.shape[:2]:
                masks = np.asarray(Image.fromarray(masks).resize(frame.shape[1::-1], Image.NEAREST))
            mask = masks == sid
            for name, data_url in zip(("full", "crop"), region_images(frame, mask)):
                (out / f"{k:02d}_{name}.jpg").write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
        else:
            Image.fromarray(frame).save(out / f"{k:02d}_full.jpg", quality=90)
        qwen = describe(url, model, frame, mask)
        results.append({"k": k, "file": path.name, "semantic_id": sid, "whole_image": bool(whole),
                        "dam": str(e.get("description", "")), "qwen": qwen})
        print(f"[{k}] sid {sid}\n  DAM : {results[-1]['dam']}\n  QWEN: {qwen}", flush=True)
    (out / "comparison.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
