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


from ragmap_adapter.responses_shim import MUST_RETRIEVE


def _kind(item: Any) -> str | None:
    return item.get("type") if isinstance(item, dict) else getattr(item, "type", None)


def _user_text(items: list[Any]) -> str:
    """The first user turn's text: the key that ties a question's create calls to its final parse."""

    for item in items:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role != "user":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if isinstance(content, str):
            return content
        parts = [p.get("text") if isinstance(p, dict) else getattr(p, "text", None) for p in content or []]
        return "\n".join(p for p in parts if p)
    return ""


class NativeResponses:
    """The real SDK, plus the two opt-in guards of ``responses_shim`` ported to the native path.

    Ours (2026-10-29, labelled deviations, both off by default; same flags as the chat shim):

    * ``must_retrieve`` -- a first ``create`` (no tool output in its input yet) whose reply calls no tool is
      re-asked (<= 2 times) with ``MUST_RETRIEVE`` appended; DAAAM's loop then gets the re-asked reply.
    * ``round_limit_evidence`` -- at the 15-iteration limit DAAAM's ``parse`` input keeps only user turns and
      ``message`` / ``reasoning`` items. With the flag, a parse whose last user turn says "Max iterations reached"
      and that carries no tool output gets the question's last complete exchange instead: the input of its last
      ``create`` (tool calls with their outputs) plus that reply's reasoning and message items (a trailing tool
      call without its output is left out, which the Responses API would reject).
    """

    def __init__(self, base_url: str, model: str, *, reasoning_effort: str | None = "medium",
                 log: list[dict[str, Any]] | None = None, timeout: float = 1800.0,
                 max_output_tokens: int | None = None, must_retrieve: bool = False,
                 round_limit_evidence: bool = False) -> None:
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
        self.must_retrieve = must_retrieve
        self.round_limit_evidence = round_limit_evidence
        self._transcripts: dict[str, list[Any]] = {}
        create = self._wrap("create", self.client.responses.create)
        parse = self._wrap("parse", self.client.responses.parse)
        self.responses = SimpleNamespace(create=self._guarded_create(create) if (must_retrieve or round_limit_evidence)
                                         else create,
                                         parse=self._guarded_parse(parse) if round_limit_evidence else parse)

    def _guarded_create(self, create: Any):
        def call(**kwargs: Any) -> Any:
            response = create(**kwargs)
            items = list(kwargs.get("input") or [])
            first_turn = not any(_kind(i) == "function_call_output" for i in items)
            if self.must_retrieve and kwargs.get("tools") and first_turn:
                for attempt in range(1, 3):
                    if any(_kind(o) == "function_call" for o in response.output or []):
                        break
                    response = create(**{**kwargs, "input": items + [{"role": "user", "content": MUST_RETRIEVE}]})
                    self.log[-1]["must_retrieve_reask"] = attempt
            if self.round_limit_evidence:
                kept = [o for o in response.output or [] if _kind(o) in ("reasoning", "message")]
                self._transcripts[_user_text(items)] = items + kept
            return response

        return call

    def _guarded_parse(self, parse: Any):
        def call(**kwargs: Any) -> Any:
            items = list(kwargs.get("input") or [])
            last = items[-1] if items else None
            last_text = _user_text([last]) if last is not None else ""
            evidence = False
            if "Max iterations reached" in last_text and not any(_kind(i) == "function_call_output" for i in items):
                transcript = self._transcripts.get(_user_text(items))
                if transcript:
                    kwargs = {**kwargs, "input": transcript + [last]}
                    evidence = True
            response = parse(**kwargs)
            if evidence:
                self.log[-1]["round_limit_evidence"] = True
            return response

        return call

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
