"""DAAAM's own NaVQA / OC-NaVQA agent (``scripts/eval_navqa.py``) on our served LLM.

    ragmap-run navqa --output /output [--protocol oc-navqa|navqa] [--annotations corrected|legacy]
                     [--seeds 0] [--questions DIR] SEQ=DSG_PATH [SEQ=DSG_PATH ...]

``RAGMAP_VLM_BASE_URL`` / ``RAGMAP_VLM_MODEL`` name the served LLM (a
``ragmap-vlm-service`` that keeps multi-turn conversations). Its client is
``responses_shim.ResponsesOverChat`` in place of ``openai.OpenAI`` -- the one
substitution; see that module for how tool calling and structured output are
served.

``--protocol oc-navqa`` (default) is upstream's evaluation unchanged:
``eval_navqa.main`` with its questions (``data/oc-navqa/questions``), its
annotation CSV (``corrected`` = ``data/oc-navqa_data.csv``, the release's;
``legacy`` = ``data/oc-navqa_data-legacy.csv``, the paper's), full-sequence
temporal filtering ``[0, current time]``, its tools and its metrics. Only
``MODEL_NAME`` (the served model), ``OUTPUT_DIR`` and ``SEEDS`` are set.

``--protocol navqa`` is OUR LABELLED RE-IMPLEMENTATION of the modification the
paper describes for the original NaVQA (arXiv 2512.00565 §4.1, Tab. 1; the code
for it is not released): (1) **clip windows**: the scene graph is filtered to the
question's NaVQA clip ``[start_time, min(end_time, current_time)]`` instead of
``[0, current_time]``; (2) **view-position answers**: a position answer is
replaced by the robot's position when it observed the answered object -- the
object of the windowed graph nearest to the agent's answer, the agent-layer
pose nearest in time to the median of that object's in-window observation times
(its first/last observation when it has no per-frame times). Questions come
from ``--questions DIR/<seq>/*.json`` in eval_navqa's question format (built
from NaVQA by RAGMAP's importer; positions in CODa ``dense_global``), and are
scored with eval_navqa's ``evaluate_output``. No CSV correction is applied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DAAAM_ROOT = Path(os.environ.get("DAAAM_ROOT", Path(__file__).resolve().parents[1]))
SCHEMA = "ragmap.daaam_navqa/v1"
CSV = {"corrected": "oc-navqa_data.csv", "legacy": "oc-navqa_data-legacy.csv"}
#: An answer farther than this from every object keeps its own position (and is flagged).
VIEW_SNAP_MAX_M = 10.0


def _load_eval(output: Path, model: str, seeds: Sequence[int], annotations: str, client: Any):
    sys.path.insert(0, str(DAAAM_ROOT / "scripts"))
    import eval_navqa as ev
    from daaam.scene_understanding import services

    ev.MODEL_NAME = model
    ev.OUTPUT_DIR = output
    ev.SEEDS = tuple(seeds)
    ev.CORRECTED_CSV = DAAAM_ROOT / "data" / CSV[annotations]
    services.create_client = lambda *a, **k: client  # the one substitution
    return ev


def _observation_times(node: Any, lo: float, hi: float) -> list[float]:
    history = (node.attributes.metadata.get() or {}).get("temporal_history") or {}
    times = [float(t) for t in history.get("timestamps") or [] if lo <= float(t) <= hi]
    if not times:
        times = [float(history[k]) for k in ("first_observed", "last_observed")
                 if history.get(k) is not None and lo <= float(history[k]) <= hi]
    return times


def view_position(sg: Any, answer: Sequence[float], lo: float, hi: float) -> dict[str, Any]:
    import spark_dsg as sdsg

    target = np.asarray(answer, dtype=float)
    best = None
    for layer in (sdsg.DsgLayers.OBJECTS, "BACKGROUND_OBJECTS"):
        for node in sg.get_layer(layer).nodes:
            distance = float(np.linalg.norm(np.asarray(node.attributes.position, dtype=float) - target))
            if best is None or distance < best[0]:
                best = (distance, node)
    agents = [(float(n.attributes.metadata.get()["timestamp"]), np.asarray(n.attributes.position, dtype=float))
              for n in sg.get_layer(2, 97).nodes]
    info: dict[str, Any] = {"object_answer": list(map(float, target))}
    if best is None or not agents or best[0] > VIEW_SNAP_MAX_M:
        info.update(view_position=list(map(float, target)), snapped=False,
                    reason="no object within %.0f m" % VIEW_SNAP_MAX_M if best else "no object/agent nodes",
                    nearest_object_m=None if best is None else best[0])
        return info
    distance, node = best
    times = _observation_times(node, lo, hi)
    when = float(np.median(times)) if times else float(np.clip(
        (node.attributes.metadata.get().get("temporal_history") or {}).get("last_observed", hi), lo, hi))
    pose_time, pose = min(agents, key=lambda a: abs(a[0] - when))
    info.update(view_position=list(map(float, pose)), snapped=True, object=str(node.id), nearest_object_m=distance,
                observation_time=when, pose_time=pose_time, observations_in_window=len(times))
    return info


def run_navqa_protocol(ev: Any, dsg_paths: dict[int, Path], questions_dir: Path, seeds: Sequence[int],
                       output: Path) -> dict[str, Any]:
    """The labelled NaVQA variant: clip windows, view-position answers (module docstring)."""
    import spark_dsg as sdsg
    from daaam.scene_understanding.config import SceneUnderstandingConfig
    from daaam.scene_understanding.services import SceneUnderstandingAgent
    from daaam.utils.evaluation import START_TIMES, preprocess_scene_graph
    from daaam.utils.logging import ConsoleLogger

    graphs, provenances = {}, {}
    for seq_id, path in dsg_paths.items():
        sg = sdsg.DynamicSceneGraph.load(str(path))
        provenances[seq_id] = ev.read_provenance(sg, path)
        graphs[seq_id] = preprocess_scene_graph(sg, START_TIMES[seq_id], path.parent / "background_objects.yaml")
    clip_handler, sentence_handler = ev.build_handlers(provenances)
    config = SceneUnderstandingConfig(model_name=ev.MODEL_NAME, available_tools=list(ev.AVAILABLE_TOOLS))
    config.tool_config.sentence_embedding_model_name = ev.SENTENCE_EMBEDDING_MODEL
    config.tool_config.clip_model_name = clip_handler.model_name
    config.tool_config.clip_backend = clip_handler.backend
    agent = SceneUnderstandingAgent(config, ConsoleLogger(), clip_handler=clip_handler, sentence_handler=sentence_handler)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = {}
    for seed in seeds:
        for seq_id, base in graphs.items():
            start = START_TIMES[seq_id]
            samples = []
            for path in sorted((questions_dir / str(seq_id)).glob("*.json")):
                samples += json.loads(path.read_text(encoding="utf-8"))["data"]
            responses = []
            for index, sample in enumerate(samples):
                current = sample["current_time"] - start
                lo, hi = sample["start_time"] - start, min(sample["end_time"] - start, current)
                print(f"\n[navqa seq{seq_id} seed{seed} {index + 1}/{len(samples)}] window [{lo:.1f}, {hi:.1f}] {sample['question']}")
                sg = base.clone()
                removed = ev.filter_scene_graph_by_time(sg, lo, hi)
                agent.update_scene_graph(sg)
                started = time.time()
                try:
                    answer, history, _ = agent.answer_query(ev.RESPONSE_TYPES[sample["type"]], sample["question"])
                except Exception as exc:  # noqa: BLE001 - recorded like eval_navqa.answer_question
                    traceback.print_exc()
                    responses.append({"id": sample["id"], "question": sample["question"], "type": sample["type"],
                                      "response": None, "ground_truth": sample["answers"],
                                      "error": {"exception": str(exc), "traceback": traceback.format_exc()},
                                      "elapsed": time.time() - started, "window": [lo, hi], "filter_removed": removed})
                    continue
                view = None
                if sample["type"] == "position" and answer.answer is not None and len(answer.answer) == 3:
                    view = view_position(sg, answer.answer, lo, hi)
                    answer = answer.model_copy(update={"answer": view["view_position"]})
                error = ev.evaluate_output(sample, answer)
                history_serialised, usage = ev.summarize_interaction(history)
                print(f"answer={answer.answer!r} error={ {k: v for k, v in error.items() if k.endswith('_error') or k == 'binary_iscorrect'} }")
                responses.append({"id": sample["id"], "question": sample["question"], "type": sample["type"],
                                  "response": {"answer": answer.answer, "reasoning": answer.reasoning},
                                  "view_position": view, "ground_truth": sample["answers"], "error": error,
                                  "elapsed": time.time() - started, "window": [lo, hi], "filter_removed": removed,
                                  "interaction_history": history_serialised, "token_usage": usage})
            path = output / f"navqa_clip_seq_{seq_id}_{seed}_{stamp}.json"
            path.write_text(json.dumps({"version": "ragmap-navqa-0.1", "protocol": "navqa (clip windows, view positions)",
                                        "metadata": {"sequence_id": seq_id, "seed_id": seed, "dsg_path": str(dsg_paths[seq_id]),
                                                     "questions_dir": str(questions_dir), "model_name": ev.MODEL_NAME},
                                        "responses": responses}, indent=2, default=str), encoding="utf-8")
            files[f"{seq_id}_{seed}"] = str(path)
    return files


def main(argv: Sequence[str]) -> int:
    from ragmap_adapter.responses_shim import ResponsesOverChat

    parser = argparse.ArgumentParser(prog="ragmap-run navqa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("oc-navqa", "navqa"), default="oc-navqa")
    parser.add_argument("--annotations", choices=tuple(CSV), default="corrected")
    parser.add_argument("--seeds", default="0", help="comma-separated (upstream: 0,1,2)")
    parser.add_argument("--questions", type=Path, default=None, help="navqa protocol: <dir>/<seq>/*.json")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("dsgs", nargs="+", help="SEQ=DSG_PATH (clustered_dsg_with_summaries.json)")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run: dict[str, Any] = {"schema": SCHEMA, "status": "running", "daaam_git_sha": os.environ.get("DAAAM_GIT_SHA"),
                           "argv": list(argv)}
    calls: list[dict[str, Any]] = []
    started = time.time()
    try:
        base_url = os.environ.get("RAGMAP_VLM_BASE_URL", "").strip()
        model = os.environ.get("RAGMAP_VLM_MODEL", "").strip()
        if not base_url or not model:
            raise ValueError("RAGMAP_VLM_BASE_URL and RAGMAP_VLM_MODEL must name the served LLM.")
        api = os.environ.get("RAGMAP_VLM_API", "chat").strip()
        if api == "responses":
            # The server speaks the Responses API natively (vLLM's gpt-oss, or OpenAI itself): the real SDK.
            from ragmap_adapter.native_responses import NativeResponses

            effort = os.environ.get("RAGMAP_VLM_REASONING_EFFORT", "medium").strip() or None
            client = NativeResponses(base_url, model, reasoning_effort=effort, log=calls,
                                     max_output_tokens=args.max_tokens if args.max_tokens > 4096 else None)
            run["llm"] = {"base_url": base_url, "model": model, "api": "responses (native openai SDK)",
                          "reasoning_effort": effort, "temperature": "server default", "seed": "none (independent samples)"}
        elif api == "chat":
            temp_env = os.environ.get("RAGMAP_VLM_TEMPERATURE", "0").strip()
            temperature = None if temp_env in ("", "default", "none") else float(temp_env)
            thinking = os.environ.get("RAGMAP_VLM_THINKING", "0").strip() in ("1", "true", "on")
            flag = lambda name: os.environ.get(name, "0").strip() in ("1", "true", "on")
            must_retrieve, round_limit = flag("RAGMAP_DAAAM_MUST_RETRIEVE"), flag("RAGMAP_DAAAM_ROUND_LIMIT_EVIDENCE")
            client = ResponsesOverChat(base_url, model, max_tokens=args.max_tokens, log=calls,
                                       temperature=temperature, thinking=thinking, must_retrieve=must_retrieve,
                                       round_limit_evidence=round_limit)
            run["llm"] = {"base_url": base_url, "model": model, "max_tokens": args.max_tokens, "thinking": thinking,
                          "temperature": "server default" if temperature is None else temperature,
                          "tool_calling": "Qwen3.5 template tool format rendered client-side (responses_shim)",
                          "structured_output": "schema in system prompt, pydantic-validated, 3 attempts",
                          "must_retrieve": must_retrieve, "round_limit_evidence": round_limit}
        else:
            raise ValueError(f"RAGMAP_VLM_API must be 'chat' or 'responses', not {api!r}")
        ev = _load_eval(output, model, seeds, args.annotations, client)
        run["protocol"] = args.protocol
        if args.protocol == "oc-navqa":
            run["annotations_csv"] = str(ev.CORRECTED_CSV)
            ev.main(args.dsgs)
        else:
            if args.questions is None:
                raise ValueError("--protocol navqa needs --questions DIR")
            dsg_paths = ev.parse_dsg_arguments(args.dsgs)
            run["result_files"] = run_navqa_protocol(ev, dsg_paths, args.questions, seeds, output)
        run["status"] = "ok"
        return 0
    except BaseException as exc:  # noqa: BLE001
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["traceback"] = traceback.format_exc()
        traceback.print_exc()
        return 1
    finally:
        run["llm_calls"] = len(calls)
        run["llm_errors"] = sum(1 for c in calls if c.get("status") != 200)
        run["parse_errors"] = sum(1 for c in calls if c.get("parse_error"))
        run["timings"] = {"total_seconds": time.time() - started}
        with open(output / "llm_calls.jsonl", "a", encoding="utf-8") as handle:
            for call in calls:
                handle.write(json.dumps(call, default=str) + "\n")
        (output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
