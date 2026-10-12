"""Read-only progress reporting from inside the DAAAM node, for the adapter's drain.

This directory is put on ``PYTHONPATH`` only for the ``ros2 launch`` the
adapter starts, and only acts in the process whose script is upstream's
``daaam_node.py`` (``RAGMAP_DAAAM_STATUS`` names the file to write). It wraps
``PipelineOrchestrator.start`` to also start one daemon thread that, every
two seconds, writes the orchestrator's queue sizes and counters to that file
(atomically). It reads state and changes nothing: no queue is touched, no
lock is held longer than upstream's own reads.

The adapter uses it to know when the bag's last frames have gone through
assignment and grounding, so that the bag's ``/clock`` is held until then
(see ``run.py``, "drain").
"""
from __future__ import annotations

import os
import sys


def _install() -> None:
    status_path = os.environ.get("RAGMAP_DAAAM_STATUS")
    if not status_path or not sys.argv or not sys.argv[0].endswith("daaam_node.py"):
        return
    import json
    import threading
    import time

    from daaam.pipeline import orchestrator as orchestrator_module

    original_start = orchestrator_module.PipelineOrchestrator.start

    def _qsize(queue: object) -> int:
        try:
            return int(queue.qsize())  # type: ignore[attr-defined]
        except (NotImplementedError, AttributeError, OSError):
            return -1

    def _report(orchestrator: object) -> None:
        tmp = status_path + ".tmp"
        while True:
            try:
                service = getattr(orchestrator, "scene_graph_service", None)
                status = {
                    "time": time.time(),
                    "frame_count": int(getattr(orchestrator, "frame_count", 0)),
                    "queues": {
                        name: _qsize(getattr(orchestrator, name))
                        for name in ("assignment_task_queue", "selected_groups_queue",
                                     "query_group_queue", "correction_queue")
                        if hasattr(orchestrator, name)
                    },
                    "pending_track_ids": len(getattr(orchestrator, "pending_track_ids", ())),
                    "prompted_track_ids": len(getattr(orchestrator, "prompted_track_ids", ())),
                    "object_labels": len(getattr(orchestrator, "object_labels", {})),
                    "corrections_stored": int(getattr(orchestrator, "_corrections_stored", 0)),
                    "corrections": len(getattr(service, "corrections", {}) or {}),
                    "output_dir": str(getattr(orchestrator, "output_dir", "")),
                }
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(status, handle)
                os.replace(tmp, status_path)
            except Exception as exc:  # noqa: BLE001 - reporting must never take the node down
                print(f"[ragmap hook] status write failed: {exc}", file=sys.stderr)
            time.sleep(2.0)

    def start(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_start(self, *args, **kwargs)
        threading.Thread(target=_report, args=(self,), name="ragmap-status", daemon=True).start()
        return result

    orchestrator_module.PipelineOrchestrator.start = start


try:
    _install()
except Exception as _exc:  # noqa: BLE001
    print(f"[ragmap hook] not installed: {_exc}", file=sys.stderr)
