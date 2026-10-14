"""``openai.OpenAI`` itself, pointed at a server that speaks the Responses API natively.

Used for the NaVQA repro ladder's gpt-oss arm (``RAGMAP_VLM_API=responses``): vLLM serves
OpenAI's open-weight reasoning model gpt-oss with its own harmony renderer, native tool
calling and ``/v1/responses`` (structured output included), so DAAAM's agent loop runs
**unchanged** -- the real OpenAI SDK, ``responses.create`` with tools and
``responses.parse`` with ``text_format`` -- and only two things are set at the client
boundary:

* ``model`` -> the served model name (DAAAM asks for ``gpt-5-mini``);
* ``reasoning={"effort": ...}`` when the call does not set one (DAAAM never does, so
  GPT-5-mini ran at OpenAI's default, ``medium``; ``RAGMAP_VLM_REASONING_EFFORT``).

No temperature and no seed are sent (DAAAM sends none; GPT-5-mini samples at its fixed
default), so the server's / model's default sampling applies and seeds 0,1,2 are three
independent samples, as upstream. With ``base_url=https://api.openai.com/v1`` and
``OPENAI_API_KEY`` set this is exactly upstream's client.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any


class NativeResponses:
    def __init__(self, base_url: str, model: str, *, reasoning_effort: str | None = "medium",
                 log: list[dict[str, Any]] | None = None, timeout: float = 1800.0,
                 max_output_tokens: int | None = None) -> None:
        from openai import OpenAI

        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        self.client = OpenAI(base_url=url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", timeout=timeout,
                             max_retries=5)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.log = log if log is not None else []
        self.responses = SimpleNamespace(create=self._wrap("create", self.client.responses.create),
                                         parse=self._wrap("parse", self.client.responses.parse))

    def _wrap(self, kind: str, method: Any):
        def call(**kwargs: Any) -> Any:
            requested = kwargs.get("model")
            kwargs["model"] = self.model
            if self.reasoning_effort and "reasoning" not in kwargs:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.max_output_tokens and "max_output_tokens" not in kwargs:
                kwargs["max_output_tokens"] = self.max_output_tokens
            started = time.time()
            entry: dict[str, Any] = {"kind": kind, "requested_model": requested, "served_model": self.model,
                                     "reasoning": kwargs.get("reasoning")}
            try:
                response = method(**kwargs)
            except Exception as exc:  # noqa: BLE001 - logged, then raised to DAAAM's handling
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000],
                             seconds=round(time.time() - started, 2))
                self.log.append(entry)
                raise
            usage = getattr(response, "usage", None)
            entry.update(status=200, seconds=round(time.time() - started, 2),
                         response_status=getattr(response, "status", None),
                         output_types=[getattr(i, "type", None) for i in getattr(response, "output", []) or []],
                         usage=usage.model_dump() if hasattr(usage, "model_dump") else None)
            if kind == "parse" and getattr(response, "output_parsed", None) is None:
                entry["parse_error"] = "output_parsed is None"
            self.log.append(entry)
            return response

        return call
