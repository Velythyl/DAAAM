"""Describe a masked region with a served multimodal LLM, in place of DAM-3B (ours, 2026-10-29).

DAM (``DescribeAnythingModel``, prompt_mode ``full+focal_crop``) sees two inputs per mask: the full
image with the mask and a "focal crop" around it (the mask's box enlarged 3x, at least 48 px), and
answers ``"Describe what you see in this region."``. A chat VLM has no mask channel, so the mask is
drawn: the full frame with the region's outline in red, and the same focal crop, also outlined. The
question is DAM's, with one sentence saying how the region is marked. The whole-image entry
(``semantic_id == -1``, an all-ones mask) gets the plain frame and ``"Describe what you see in this
image."``.

Greedy (temperature 0), thinking off, ``max_tokens`` 196 (DAAAM's cap on CODa, where the full-image
description is on), so a build is deterministic up to batching. Deviations from DAM: drawn outline
instead of a mask channel; greedy instead of DAM's temperature 0.2 / top_p 0.9 sampling.
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

REGION_PROMPT = ("The first image is the full scene with one region outlined in red; the second image is a close-up "
                 "around the same region, also outlined. Describe what you see in this region.")
IMAGE_PROMPT = "Describe what you see in this image."
OUTLINE = (255, 0, 0)


def outline(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Boolean map of the mask's border, ``width`` pixels thick (numpy only)."""

    m = mask.astype(bool)
    inner = m.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        inner = shrunk
    return m & ~inner


def focal_box(mask: np.ndarray, scale: float = 3.0, minimum: int = 48) -> tuple[int, int, int, int]:
    """DAM's focal crop: the mask's box enlarged ``scale`` times (at least ``minimum`` px), clipped to the image."""

    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    bw = max((xs.max() - xs.min() + 1) * scale, minimum)
    bh = max((ys.max() - ys.min() + 1) * scale, minimum)
    x0, x1 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    y0, y1 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    return x0, y0, x1, y1


def _jpeg_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def region_images(image: np.ndarray, mask: np.ndarray) -> list[str]:
    """[full frame with the region outlined, focal crop with the region outlined] as data URLs."""

    drawn = image.copy()
    drawn[outline(mask)] = OUTLINE
    x0, y0, x1, y1 = focal_box(mask)
    crop = drawn[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 384:  # DAM resizes its inputs to 384; upsample small crops for legibility
        factor = 384 / min(crop.shape[:2])
        crop = np.asarray(Image.fromarray(crop).resize((int(crop.shape[1] * factor), int(crop.shape[0] * factor))))
    return [_jpeg_url(drawn), _jpeg_url(crop)]


def describe(base_url: str, model: str, image: np.ndarray, mask: np.ndarray | None, *, max_tokens: int = 196,
             timeout: float = 600.0, log: list[dict[str, Any]] | None = None) -> str:
    """One description; ``mask=None`` (or an all-ones mask) describes the whole image."""

    whole = mask is None or bool(np.asarray(mask).all())
    urls = [_jpeg_url(image)] if whole else region_images(image, np.asarray(mask).astype(bool))
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": IMAGE_PROMPT if whole else REGION_PROMPT})
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    started = time.time()
    for attempt in range(5):
        try:
            response = requests.post(base_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code in (502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
        text = (response.json()["choices"][0]["message"]["content"] or "").strip()
        if log is not None:
            log.append({"whole": whole, "seconds": round(time.time() - started, 2), "chars": len(text)})
        return text
    raise RuntimeError(f"{base_url} did not answer")
