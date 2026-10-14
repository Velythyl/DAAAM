"""``OpenAI().responses`` as DAAAM's scene-understanding agent uses it, served by our
Qwen3.5 chat-completions endpoint (``ragmap-vlm-service``).

DAAAM's agent (``daaam.scene_understanding.services.SceneUnderstandingAgent.
_answer_query_openai``) makes two kinds of call on the OpenAI Responses API:

* ``responses.create(model, instructions, input, tools, tool_choice="auto")`` --
  one turn of its tool loop, native function calling;
* ``responses.parse(model, instructions, input, text_format=<pydantic model>)``
  -- the final answer, OpenAI structured output.

Our service has neither (it speaks ``/v1/chat/completions``, text in, text out,
greedy; the shim also sends ``temperature: 0``, so a vLLM OpenAI server, whose
default is 1.0, decodes greedily too). This shim keeps the agent's loop, prompts,
tools and parsing untouched and translates at the client boundary:

**Tool calling = Qwen3.5's own tool format, rendered client-side.** The model's
chat template (``chat_template.jinja`` of Qwen/Qwen3.5-35B-A3B-FP8 @ 9d1823d)
renders ``tools`` as a ``# Tools`` block at the head of the system turn, an
assistant's tool calls as ``<tool_call><function=NAME><parameter=K>V</parameter>
</function></tool_call>`` and tool results as a user turn of
``<tool_response>...</tool_response>`` blocks. The service does not accept
``tools`` or ``tool`` messages, so the shim writes exactly those strings itself
(``TOOLS_PREAMBLE`` is copied verbatim from the template) and parses the model's
``<tool_call>`` blocks back into Responses ``function_call`` items, as vLLM's
``qwen3_coder`` parser does. The tokens the model sees are the ones native tool
calling would give it. Needs a service that keeps multi-turn conversations
(``assistant`` messages; the query-side branch's ``_resolve_turns``).

**Structured output = the JSON schema in the system prompt** (as
``summarize.py`` does for the region summaries), the reply validated by the
same pydantic model; up to ``parse_attempts`` tries, each one logged. OpenAI's
server-side constraint cannot fail; ours can, and a reply that never parses
raises, which ``eval_navqa.answer_question`` records as a failed question.

Thinking is off (``enable_thinking=false``, the OL2.5D setting of this model);
gpt-5-mini reasons by default. Token usage is not reported by the service, so
``usage`` is ``None`` (``eval_navqa.summarize_interaction`` then counts nothing).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Sequence

import requests

from ragmap_adapter.summarize import unwrap_schema_echo

#: Verbatim from the Qwen3.5 chat template (the ``tools`` branch of the system turn).
TOOLS_PREAMBLE_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_PREAMBLE_TAIL = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)
SCHEMA_INSTRUCTION = "\n\nRespond with a single JSON object and nothing else, matching this JSON schema:\n{schema}"

_TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ShimError(RuntimeError):
    pass


def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A Responses-API function tool in the chat-completions shape the template is written for."""
    if tool.get("type") != "function":
        raise ShimError(f"Only function tools are supported, got {tool.get('type')!r}.")
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": tool.get("parameters", {})}}


def tools_block(tools: Sequence[dict[str, Any]]) -> str:
    # The template's `tool | tojson` (transformers: json.dumps, ensure_ascii=False, default separators).
    return TOOLS_PREAMBLE_HEAD + "".join("\n" + json.dumps(_chat_tool(t), ensure_ascii=False) for t in tools) \
        + TOOLS_PREAMBLE_TAIL


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or ():
        block = block if isinstance(block, dict) else vars(block)
        if block.get("type") in ("input_text", "output_text", "text"):
            parts.append(str(block.get("text") or ""))
        else:
            raise ShimError(f"Unsupported content block {block.get('type')!r} (text only).")
    return "".join(parts)


def _render_call(name: str, arguments: str) -> str:
    args = json.loads(arguments) if arguments else {}
    body = "".join(
        f"<parameter={key}>\n"
        f"{json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}\n</parameter>\n"
        for key, value in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def to_chat_messages(system: str, items: Sequence[Any]) -> list[dict[str, str]]:
    """Responses ``input`` items -> chat turns rendered as Qwen's template renders them."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    pending_assistant: list[Any] = []
    pending_tool: list[str] = []

    def flush() -> None:
        if pending_assistant:
            text = "".join(_text(i.content) for i in pending_assistant if i.type == "message").strip()
            calls = [_render_call(i.name, i.arguments) for i in pending_assistant if i.type == "function_call"]
            content = text
            if calls:
                content = (text + "\n\n" if text else "") + "\n".join(calls)
            messages.append({"role": "assistant", "content": content})
            pending_assistant.clear()
        if pending_tool:
            messages.append({"role": "user", "content": "\n".join(
                f"<tool_response>\n{out}\n</tool_response>" for out in pending_tool)})
            pending_tool.clear()

    for item in items:
        if isinstance(item, dict):
            if item.get("type") == "function_call_output":
                if pending_assistant:
                    flush()
                pending_tool.append(str(item.get("output", "")))
                continue
            role = item.get("role")
            if role not in ("user", "system", "developer"):
                raise ShimError(f"Unsupported input item {item!r:.200}.")
            flush()
            if role == "user":
                messages.append({"role": "user", "content": _text(item.get("content"))})
            else:
                messages[0]["content"] += "\n\n" + _text(item.get("content"))
            continue
        kind = getattr(item, "type", None)
        if kind == "reasoning":
            continue
        if kind not in ("message", "function_call"):
            raise ShimError(f"Unsupported output item type {kind!r}.")
        if pending_tool:
            flush()
        pending_assistant.append(item)
    flush()
    return messages


def _coerce(value: str, schema: dict[str, Any]) -> Any:
    """A parameter's text as the schema's type (vLLM's qwen3_coder parser does the same)."""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    if "string" in kinds and len(kinds) == 1:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_tool_calls(text: str, tools: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    props = {t["name"]: (t.get("parameters") or {}).get("properties") or {} for t in tools}
    calls = []
    for match in _TOOL_CALL.finditer(text):
        name = match.group(1).strip()
        args = {key: _coerce(value, props.get(name, {}).get(key, {}))
                for key, value in _PARAM.findall(match.group(2))}
        calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
    head = text.split("<tool_call>", 1)[0].strip() if calls else text.strip()
    return head, calls


def json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in the reply: {text[:200]!r}")
    return text[start:end + 1]


def _message_item(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", role="assistant", id=f"msg_{uuid.uuid4().hex}",
                           content=[SimpleNamespace(type="output_text", text=text)])


class ResponsesOverChat:
    """Duck-types ``openai.OpenAI`` for ``responses.create`` / ``responses.parse``."""

    def __init__(self, base_url: str, model: str, *, max_tokens: int = 4096, timeout: float = 900.0,
                 parse_attempts: int = 3, log: list[dict[str, Any]] | None = None,
                 temperature: float | None = 0.0, thinking: bool = False) -> None:
        # temperature None = the server's / model's default sampling; thinking = Qwen3.5's thinking mode
        # (the reasoning is split off by vLLM's reasoning parser or stripped below; never fed back).
        self.temperature = temperature
        self.thinking = thinking
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.parse_attempts = parse_attempts
        self.log = log if log is not None else []
        self.responses = SimpleNamespace(create=self._create, parse=self._parse)

    def _post(self, kind: str, requested: str, messages: list[dict[str, str]]) -> str:
        started = time.time()
        entry: dict[str, Any] = {"kind": kind, "requested_model": requested, "served_model": self.model,
                                 "messages": messages}
        for attempt in range(5):
            try:
                body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                        "chat_template_kwargs": {"enable_thinking": self.thinking}}
                if self.temperature is not None:
                    body["temperature"] = float(self.temperature)
                response = requests.post(f"{self.base_url}/v1/chat/completions", timeout=self.timeout, json=body)
            except requests.RequestException as exc:
                entry.setdefault("transport_errors", []).append(str(exc))
                time.sleep(5 * (attempt + 1))
                continue
            if response.status_code in (502, 503, 504):
                time.sleep(float(response.headers.get("Retry-After", 2 * (attempt + 1))))
                continue
            break
        else:
            entry["status"] = "unreachable"
            self.log.append(entry)
            raise ShimError(f"{self.base_url} did not answer: {entry.get('transport_errors')}")
        entry.update(status=response.status_code, seconds=round(time.time() - started, 2))
        if response.status_code != 200:
            entry["error"] = response.text[:2000]
            self.log.append(entry)
            raise ShimError(f"chat/completions {response.status_code}: {response.text[:500]}")
        payload = response.json()
        message = payload["choices"][0]["message"]
        text = message["content"] or ""
        entry["reply"] = text
        entry["reasoning"] = message.get("reasoning_content") or message.get("reasoning")
        entry["finish_reason"] = payload["choices"][0].get("finish_reason")
        entry["usage"] = payload.get("usage")
        self.log.append(entry)
        return _THINK.sub("", text).strip()

    def _create(self, *, model: str, instructions: str, input: list[Any], tools: list[dict[str, Any]],  # noqa: A002
                tool_choice: str = "auto", **_: Any) -> SimpleNamespace:
        system = tools_block(tools) + ("\n\n" + instructions.strip() if instructions and instructions.strip() else "")
        text = self._post("create", model, to_chat_messages(system, input))
        head, calls = parse_tool_calls(text, tools)
        # Native tool calling (OpenAI's, or a constrained server) can only name a declared
        # tool. Qwen sometimes "calls" an undeclared one to give its answer (e.g. `answer`);
        # such a call is kept as the reply's text, so the agent goes on to its final parse.
        known = {t["name"] for t in tools}
        unknown = [c for c in calls if c["name"] not in known]
        if unknown:
            calls = [c for c in calls if c["name"] in known]
            head = (head + "\n\n" if head else "") + "\n".join(c["arguments"] for c in unknown)
            self.log[-1]["undeclared_tool_calls"] = [c["name"] for c in unknown]
        output: list[Any] = []
        if head:
            output.append(_message_item(head))
        for call in calls:
            output.append(SimpleNamespace(type="function_call", name=call["name"], arguments=call["arguments"],
                                          call_id=f"call_{uuid.uuid4().hex[:24]}", id=f"fc_{uuid.uuid4().hex}"))
        return SimpleNamespace(output=output, output_text=head, usage=None, model=self.model)

    def _parse(self, *, model: str, instructions: str, input: list[Any], text_format: Any,  # noqa: A002
               **_: Any) -> SimpleNamespace:
        schema = json.dumps(text_format.model_json_schema())
        system = (instructions or "").strip() + SCHEMA_INSTRUCTION.format(schema=schema)
        messages = to_chat_messages(system, input)
        errors = []
        for _attempt in range(self.parse_attempts):
            text = self._post("parse", model, messages)
            try:
                parsed = text_format.model_validate(unwrap_schema_echo(json.loads(json_object(text)), text_format))
            except Exception as exc:  # noqa: BLE001 - pydantic or JSON; retried, then raised
                errors.append(f"{type(exc).__name__}: {exc}"[:300])
                self.log[-1]["parse_error"] = errors[-1]
                continue
            return SimpleNamespace(output_parsed=parsed, output=[_message_item(text)], usage=None,
                                   status="completed", model=self.model)
        raise ShimError(f"No reply matched {text_format.__name__} in {self.parse_attempts} attempts: {errors}")
