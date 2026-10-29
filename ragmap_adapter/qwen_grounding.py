"""DAAAM's grounding worker with a served VLM describing the regions instead of DAM-3B (ours, 2026-10-29).

Only the describer changes. ``DAMGroundingWorkerMultiImage`` (batching to ``multi_image_min_n_masks``,
the whole-image entry, select-frame CLIP features, saving, corrections) runs as upstream; its module-level
``DAMAgentPanoptic`` is replaced, in the worker process that imports this module, by ``QwenAgentPanoptic``,
which has the same constructor and ``query_multi_image_multi_mask`` and asks ``qwen_describer.describe`` for
every mask of a batch concurrently. Served model: ``RAGMAP_DESCRIBER_BASE_URL`` / ``RAGMAP_DESCRIBER_MODEL``.

``hook/sitecustomize.py`` makes the grounding service start ``QwenGroundingWorkerMultiImage`` when
``RAGMAP_DAAAM_DESCRIBER=qwen``; the worker is a spawned process that unpickles its target by reference,
so importing this module there is what installs the replacement.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from daaam.grounding.workers import dam_grounding
from ragmap_adapter.qwen_describer import describe

_log = logging.getLogger("ragmap.qwen_grounding")


class QwenAgentPanoptic:
    """Stands in for ``DAMAgentPanoptic``; the DAM arguments are accepted and ignored."""

    def __init__(self, model_path: str | None = None, conv_mode: str | None = None,
                 prompt_mode: str | None = None, **_: Any) -> None:
        self.base_url = os.environ["RAGMAP_DESCRIBER_BASE_URL"]
        self.model = os.environ["RAGMAP_DESCRIBER_MODEL"]
        self.workers = int(os.environ.get("RAGMAP_DESCRIBER_CONCURRENCY", "16"))
        self.calls: list[dict[str, Any]] = []
        _log.info("Qwen describer: %s @ %s (replaces %s)", self.model, self.base_url, model_path)

    def query_multi_image_multi_mask(self, image_mask_pairs: list[tuple[Any, list[Any]]], query: str = "",
                                     temperature: float = 0.0, top_p: float = 1.0,
                                     max_new_tokens: int = 196, **_: Any) -> list[list[str]]:
        jobs = []
        for i, (image, masks) in enumerate(image_mask_pairs):
            frame = np.asarray(image.convert("RGB"))
            for j, mask in enumerate(masks):
                m = np.asarray(mask) > 0
                jobs.append((i, j, frame, None if m.all() else m))
        out: list[list[str]] = [[""] * len(masks) for _, masks in image_mask_pairs]

        def run(job):
            i, j, frame, m = job
            try:
                return i, j, describe(self.base_url, self.model, frame, m, max_tokens=max_new_tokens, log=self.calls)
            except Exception as exc:  # noqa: BLE001 - an empty answer becomes "unknown" and is re-prompted upstream
                _log.warning("describer failed: %s", exc)
                return i, j, ""

        with ThreadPoolExecutor(self.workers) as pool:
            for i, j, text in pool.map(run, jobs):
                out[i][j] = text
        return out


dam_grounding.DAMAgentPanoptic = QwenAgentPanoptic


class QwenGroundingWorkerMultiImage(dam_grounding.DAMGroundingWorkerMultiImage):
    """Upstream's multi-image worker; defined here so the spawned process imports this module."""
