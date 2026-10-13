"""DAAAM's region summaries (``scripts/summarize_regions.py``) on our served LLM.

    ragmap-run summarize --input /input --output /output [key=value ...]

INPUT ``/input``: the object-mapping container's OUTPUT (``daaam/`` with
``clustered_dsg.json``, ``corrections.yaml``, ``background_objects.yaml``;
``regions.jsonl``). OUTPUT ``/output``: ``daaam/`` (those three files plus
upstream's ``region_summaries.yaml`` and ``clustered_dsg_with_summaries.json``,
the pair DAAAM's NaVQA agent reads), ``areas.jsonl`` (RAGMAP's area contract,
children as the object mapper's raw ids) and ``run.json``.

**What runs.** Upstream's ``summarize_regions.main`` unmodified -- its
farthest-first sampling of ``n_samples`` (20) objects per region over the
concatenated CLIP + sentence-t5 features (``feature_key=combined``), its
prompt templates (``config/prompt_templates/region_summary_*.txt``), its three
attempts per region -- with one substitution: the ``OpenAI`` client it builds
is replaced by ``ChatCompletionsClient``, which serves the one call upstream
makes, ``client.responses.parse(model, input, text_format=RegionSummary)``,
through the OpenAI-compatible ``/v1/chat/completions`` of our VLM service
(``RAGMAP_VLM_BASE_URL``, model ``RAGMAP_VLM_MODEL``). gpt-5-nano's structured
output is a server-side JSON-schema constraint our service does not have, so
the schema is stated in the system prompt (``SCHEMA_INSTRUCTION``) and the
reply parsed with the same pydantic model; a reply that does not parse is an
exception, which upstream's own retry loop counts as a failed attempt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import requests
import yaml

DAAAM_ROOT = Path(os.environ.get("DAAAM_ROOT", Path(__file__).resolve().parents[1]))
SCHEMA = "ragmap.daaam_regions/v1"
#: Appended to upstream's system prompt, standing in for OpenAI structured output.
SCHEMA_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else, matching this JSON schema:\n{schema}"
)
COPIED = ("clustered_dsg.json", "corrections.yaml", "background_objects.yaml")


class ChatCompletionsClient:
    """``OpenAI()`` as far as ``summarize_regions.py`` uses it."""

    def __init__(self, base_url: str, model: str, max_tokens: int, timeout: float, log: list[dict[str, Any]]):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.log = log
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, *, model: str, input: list[dict[str, str]], text_format: Any) -> Any:  # noqa: A002 - upstream's name
        schema = json.dumps(text_format.model_json_schema())
        messages = [dict(message) for message in input]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"] + SCHEMA_INSTRUCTION.format(schema=schema)
        else:
            messages.insert(0, {"role": "system", "content": SCHEMA_INSTRUCTION.format(schema=schema).strip()})
        started = time.time()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages, "max_tokens": self.max_tokens},
            timeout=self.timeout,
        )
        entry: dict[str, Any] = {"requested_model": model, "served_model": self.model,
                                 "seconds": round(time.time() - started, 2), "status": response.status_code}
        self.log.append(entry)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"] or ""
        entry["reply"] = text
        parsed = text_format.model_validate(unwrap_schema_echo(json.loads(_json_object(text)), text_format))
        return SimpleNamespace(status="completed", output_parsed=parsed, output=[], incomplete_details=None)


def unwrap_schema_echo(value: Any, text_format: Any) -> Any:
    """``{"properties": {<the fields>}}`` -> ``{<the fields>}``.

    With the JSON schema in its prompt, Qwen3.5 sometimes answers with the
    instance nested under the schema's ``properties`` key (room1: 1 of 2
    regions failed all 3 attempts this way). OpenAI's server-side structured
    output cannot produce that shape, so it is unwrapped rather than counted as
    a failed attempt; any other shape is validated unchanged.
    """
    required = set(text_format.model_json_schema().get("required") or ())
    if (isinstance(value, dict) and set(value) == {"properties"} and isinstance(value["properties"], dict)
            and required and required <= set(value["properties"])):
        return value["properties"]
    return value


def _json_object(text: str) -> str:
    """The reply's JSON object: a bare object, or one inside a ```json fence."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in the reply: {text[:200]!r}")
    return text[start:end + 1]


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="ragmap-run summarize")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    settings = {"n_samples": 20, "feature_key": "combined", "max_tokens": 2048, "timeout_s": 900.0}
    for item in args.overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in settings:
            raise SystemExit(f"Unknown override {item!r}; known: {sorted(settings)}")
        settings[key] = type(settings[key])(value)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {"schema": SCHEMA, "status": "running", "daaam_git_sha": os.environ.get("DAAAM_GIT_SHA"),
                           "settings": settings}
    calls: list[dict[str, Any]] = []
    started = time.time()
    try:
        base_url = os.environ.get("RAGMAP_VLM_BASE_URL", "").strip()
        model = os.environ.get("RAGMAP_VLM_MODEL", "").strip()
        if not base_url or not model:
            raise ValueError("RAGMAP_VLM_BASE_URL and RAGMAP_VLM_MODEL must name the served LLM.")
        source = args.input / "daaam"
        data = output / "daaam"
        data.mkdir(parents=True, exist_ok=True)
        for name in COPIED:
            if (source / name).is_file():
                shutil.copy2(source / name, data / name)
        if not (data / "clustered_dsg.json").is_file():
            raise FileNotFoundError(f"{source / 'clustered_dsg.json'} is missing; run the map first.")
        sys.path.insert(0, str(DAAAM_ROOT / "scripts"))
        import summarize_regions

        client = ChatCompletionsClient(base_url, model, int(settings["max_tokens"]), float(settings["timeout_s"]), calls)
        summarize_regions.OpenAI = lambda *a, **k: client  # the one substitution
        try:
            summarize_regions.main.main(
                ["--data-dir", str(data), "--model-name", model, "--n-samples", str(settings["n_samples"]),
                 "--feature-key", str(settings["feature_key"]), "--scene-graph-file", "clustered_dsg.json"],
                standalone_mode=False,
            )
        finally:
            (output / "llm_calls.jsonl").write_text("".join(json.dumps(call) + "\n" for call in calls), encoding="utf-8")
        summaries = yaml.safe_load((data / "region_summaries.yaml").read_text(encoding="utf-8")) or {}
        by_node = {str(entry["region_id"]): entry for entry in summaries.get("region_summaries") or []}
        regions = [json.loads(line) for line in (args.input / "regions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        areas = []
        failed = 0
        for region in regions:
            summary = by_node.get(region["node"])
            if summary is None:
                failed += 1
            areas.append({
                "id": region["id"],
                # A region upstream could not summarise keeps its node name.
                "label": (summary or {}).get("region_label") or region["node"],
                "summary": (summary or {}).get("region_description"),
                "centroid": region["centroid"],
                "level": 1,
                "parent": None,
                "children": region["children"],
                "extra": {"node": region["node"], "reasoning": (summary or {}).get("reasoning"),
                          "num_objects": (summary or {}).get("num_objects"),
                          "num_sampled": (summary or {}).get("num_sampled"),
                          "summarized": summary is not None, "num_places": region.get("num_places")},
            })
        (output / "areas.jsonl").write_text("".join(json.dumps(area) + "\n" for area in areas), encoding="utf-8")
        run["counts"] = {"regions": len(regions), "summarized": len(regions) - failed, "unsummarized": failed,
                         "llm_calls": len(calls), "llm_errors": sum(call.get("status") != 200 for call in calls)}
        run["llm"] = {"base_url": base_url, "model": model, "max_tokens": settings["max_tokens"],
                      "decoding": "service default (greedy)", "structured_output": "schema in system prompt"}
        run["status"] = "ok"
        return 0
    except BaseException as exc:  # noqa: BLE001
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["traceback"] = traceback.format_exc()
        traceback.print_exc()
        return 1
    finally:
        run["timings"] = {"total_seconds": time.time() - started}
        (output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
