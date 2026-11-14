#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

A long-running program (Python 3.13, standard library only). It reads the
trial brief at $MAKEFUN_PROMPT_FILE, probes every provider's catalog route,
registers its port and catalog with the env at $MAKEFUN_ENV_URL, serves the
env's client as an OpenAI server (POST /v1/chat/completions) and an Anthropic
one (POST /v1/messages), in JSON and in server-sent events, and exits 0 when
GET $MAKEFUN_ENV_URL/work says the workload is done.

Routing: for each request, the free models that meet its requirements are
ranked by the quality table's rating (best first), then by how little their
provider has been used so far (calls plus a penalty per refusal), then by
the brief's provider order. A provider that refuses (402, 429 or 5xx) is
skipped for the rest of that request and the request is retried on the next
candidate. Everything is a deterministic function of the brief, the requests
and the providers' answers: no wall clock, no unseeded randomness, so a
replay leaves the same state export.

`python3 main.py --list` prints the catalog document and exits 0.
"""

import json
import os
import sys
import threading
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 30
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# ---------------------------------------------------------------- brief ---

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except Exception:
                pass
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(v.get(key)) != "":
            base = string_of(v.get(key))
            break
    return {
        "id": pid,
        "name": string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": string_of(v.get("key_env")),
        "key": string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = string_of(v.get("model"))
    provider = string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = string_of(v.get("id"))
    if rid == "":
        return None
    req = v.get("requires") if is_record(v.get("requires")) else {}
    out_req = {
        "tools": req.get("tools") is True,
        "vision": req.get("vision") is True,
        "json": req.get("json") is True,
    }
    mc = req.get("min_context")
    if isinstance(mc, (int, float)):
        out_req["min_context"] = mc
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out_req,
    }


def read_brief(path):
    """Returns (directory, ratings, workload, error)."""
    directory = {"schema": PROVIDERS_SCHEMA, "providers": []}
    if path == "":
        return directory, None, None, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        return directory, None, None, "cannot read %s: %s" % (path, e)
    docs = documents_in(text)
    dir_doc = None
    for d in docs:
        if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list):
            dir_doc = d
            break
    if dir_doc is None:
        return directory, None, None, "the brief holds no %s document" % PROVIDERS_SCHEMA
    providers = []
    for v in dir_doc["providers"]:
        p = _provider_of(v)
        if p is not None:
            providers.append(p)
    ratings = None
    for d in docs:
        if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list):
            rows = []
            for v in d["ratings"]:
                r = _rating_of(v)
                if r is not None:
                    rows.append(r)
            best = []
            if isinstance(d.get("best"), list):
                for v in d["best"]:
                    if is_record(v) and string_of(v.get("id")) != "":
                        best.append({"id": string_of(v.get("id")),
                                     "rating_milli": number_or_none(v.get("rating_milli")) or 0})
            ratings = {"schema": RATINGS_SCHEMA, "ratings": rows, "best": best}
            break
    workload = None
    for d in docs:
        if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list):
            rows = []
            for v in d["manifest"]:
                w = _workload_request_of(v)
                if w is not None:
                    rows.append(w)
            workload = {"schema": WORKLOAD_SCHEMA, "manifest": rows}
            break
    return {"schema": PROVIDERS_SCHEMA, "providers": providers}, ratings, workload, None


# -------------------------------------------------------------- catalog ---

def join_base(env_url, base):
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers=None, body=None, timeout=5):
    """One HTTP request. Returns (status, headers_dict, body_bytes). Raises on
    connection failure. Does not follow redirects (the env never sends any)."""
    parts = urlparse(url)
    conn_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    conn = conn_cls(host, port, timeout=timeout)
    try:
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = string_of(v.get("name")).removeprefix("models/")
    else:
        mid = string_of(v.get("id"))
    if mid == "":
        return None
    context = number_or_none(v.get("context_length"))
    if context is None:
        context = number_or_none(v.get("context_window"))
    if context is None:
        context = number_or_none(v.get("inputTokenLimit"))
    display = string_of(v.get("display_name")) or string_of(v.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": bool_or_none(v.get("free")) if v.get("free") is not None else _zero_priced(v),
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    out = []
    for v in lst:
        m = _model_in(wire, v)
        if m is not None:
            out.append(m)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _hdrs, data = http_request("GET", url, headers=headers, timeout=timeout)
        if status < 200 or status >= 300:
            return {"provider": provider, "ok": False, "status": status, "models": [],
                    "error": "GET %s answered HTTP %d" % (url, status)}
        try:
            body = json.loads(data.decode("utf-8"))
        except Exception as e:
            return {"provider": provider, "ok": False, "status": status, "models": [],
                    "error": "GET %s answered non-JSON: %s" % (url, e)}
        return {"provider": provider, "ok": True, "status": status,
                "models": _models_in(provider["wire"], body), "error": None}
    except Exception as e:
        return {"provider": provider, "ok": False, "status": 0, "models": [],
                "error": "GET %s failed: %s" % (url, e)}


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for r in ratings["ratings"]:
        if r["provider"] == provider_id and r["model"] == model_id:
            return r
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None and row["context_length"] is not None
                           else model["context_length"]),
        "tools": (row["tools"] if row is not None else model["tools"]),
        "vision": (row["vision"] if row is not None else model["vision"]),
        "json": (row["json"] if row is not None else model["json"]),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps["context_length"] is None or caps["context_length"] < mc):
        return False
    return True


def build_catalog(probes, ratings):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row), "context_length": ctx})
        providers.append({"id": probe["provider"]["id"], "base_url": probe["provider"]["base"],
                          "key_env": probe["provider"]["key_env"], "wire": probe["provider"]["wire"],
                          "models": models})
    return {"providers": providers}


# ------------------------------------------------------- wire translation ---

def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(parts)


def part_of(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def number_or_zero(value):
    return value if isinstance(value, (int, float)) else 0


def stop_reason_of_openai(finish):
    """OpenAI finish_reason -> Anthropic stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def finish_reason_of_anthropic(stop):
    """Anthropic stop_reason -> OpenAI finish_reason."""
    if stop == "max_tokens":
        return "length"
    if stop == "tool_use":
        return "tool_calls"
    if stop == "refusal":
        return "content_filter"
    return "stop"


def finish_reason_of_gemini(reason, has_tool_calls):
    if has_tool_calls:
        return "tool_calls"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in ("SAFETY", "RECITATION"):
        return "content_filter"
    return "stop"


def stop_reason_of_gemini(reason, has_tool_calls):
    if has_tool_calls:
        return "tool_use"
    if reason == "MAX_TOKENS":
        return "max_tokens"
    if reason in ("SAFETY", "RECITATION"):
        return "refusal"
    return "end_turn"


def anthropic_to_openai(body, model, json_mode=False):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    if json_mode:
        out["response_format"] = {"type": "json_object"}
    return out


def openai_to_anthropic(body, model):
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
            "input": arguments_of(call["function"].get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": "msg_" + (body.get("id") if isinstance(body.get("id"), str) else "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of_openai(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": number_or_zero(usage.get("prompt_tokens")),
                  "output_tokens": number_or_zero(usage.get("completion_tokens"))},
    }


def _gemini_image_part(part):
    """One OpenAI content part as a Gemini part, or None."""
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        return {"text": part["text"]}
    if part.get("type") == "image_url" and is_record(part.get("image_url")):
        url = part["image_url"].get("url")
        if isinstance(url, str) and url.startswith("data:"):
            try:
                header, b64 = url[5:].split(",", 1)
                mime = header.split(";")[0] or "image/png"
                return {"inlineData": {"mimeType": mime, "data": b64}}
            except ValueError:
                return None
        if isinstance(url, str) and url != "":
            return {"fileData": {"mimeType": "image/png", "fileUri": url}}
    return None


def openai_to_gemini(body, model, json_mode=False):
    """An OpenAI Chat Completions request as a Gemini generateContent request."""
    contents = []
    system_text = ""
    call_names = {}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for msg in messages:
        if not is_record(msg):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_text = content if isinstance(content, str) else text_of(content)
            continue
        if role == "tool":
            # Function responses go into the previous model turn when possible;
            # otherwise they form their own user turn.
            name = call_names.get(msg.get("tool_call_id"), "")
            resp = {"functionResponse": {"name": name, "response": {"output": content if isinstance(content, str) else json.dumps(content)}}}
            if contents and contents[-1].get("role") == "model":
                contents[-1]["parts"].append(resp)
            else:
                contents.append({"role": "user", "parts": [resp]})
            continue
        grole = "model" if role == "assistant" else "user"
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for p in content:
                if is_record(p):
                    gp = _gemini_image_part(p)
                    if gp is not None:
                        parts.append(gp)
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tc in msg["tool_calls"]:
                if not is_record(tc) or not is_record(tc.get("function")):
                    continue
                fn = tc["function"]
                name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                if isinstance(tc.get("id"), str):
                    call_names[tc["id"]] = name
                parts.append({"functionCall": {"name": name, "args": arguments_of(fn.get("arguments"))}})
        if parts:
            contents.append({"role": grole, "parts": parts})
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for t in body["tools"]:
            if not is_record(t) or not is_record(t.get("function")):
                continue
            fn = t["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decl = {"name": fn["name"]}
            if isinstance(fn.get("description"), str):
                decl["description"] = fn["description"]
            if is_record(fn.get("parameters")):
                decl["parameters"] = fn["parameters"]
            decls.append(decl)
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    cfg = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        cfg["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        cfg["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        cfg["topP"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        cfg["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        cfg["stopSequences"] = [s for s in stop if isinstance(s, str)]
    if json_mode or (is_record(body.get("response_format")) and
                    body["response_format"].get("type") in ("json_object", "json_schema")):
        cfg["responseMimeType"] = "application/json"
    if cfg:
        out["generationConfig"] = cfg
    return out


def anthropic_to_gemini(body, model, json_mode=False):
    """An Anthropic Messages request as a Gemini generateContent request."""
    contents = []
    system_text = text_of(body.get("system"))
    call_names = {}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for msg in messages:
        if not is_record(msg):
            continue
        assistant = msg.get("role") == "assistant"
        grole = "model" if assistant else "user"
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif btype == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        parts.append({"inlineData": {
                            "mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                            "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"mimeType": "image/png", "fileUri": source["url"]}})
                elif btype == "tool_use":
                    name = block.get("name") if isinstance(block.get("name"), str) else ""
                    if isinstance(block.get("id"), str):
                        call_names[block["id"]] = name
                    parts.append({"functionCall": {"name": name, "args": block.get("input") if is_record(block.get("input")) else {}}})
                elif btype == "tool_result":
                    name = call_names.get(block.get("tool_use_id"), "")
                    parts.append({"functionResponse": {"name": name, "response": {"output": tool_result_text(block.get("content"))}}})
        if parts:
            contents.append({"role": grole, "parts": parts})
    out = {"contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decl = {"name": raw["name"]}
            if isinstance(raw.get("description"), str):
                decl["description"] = raw["description"]
            if is_record(raw.get("input_schema")):
                decl["parameters"] = raw["input_schema"]
            decls.append(decl)
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    cfg = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        cfg["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        cfg["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        cfg["topP"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        cfg["stopSequences"] = [s for s in body["stop_sequences"] if isinstance(s, str)]
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    if cfg:
        out["generationConfig"] = cfg
    return out


def gemini_to_openai(body, model, request_seq):
    """A Gemini generateContent answer as a Chat Completions object."""
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    cand = candidates[0] if candidates and is_record(candidates[0]) else {}
    content = cand.get("content") if is_record(cand.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    texts = []
    tool_calls = []
    for part in parts:
        if not is_record(part):
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        if is_record(part.get("functionCall")):
            fc = part["functionCall"]
            tool_calls.append({
                "id": "call_gemini_%d" % len(tool_calls),
                "type": "function",
                "function": {
                    "name": fc.get("name") if isinstance(fc.get("name"), str) else "",
                    "arguments": json.dumps(fc.get("args") if is_record(fc.get("args")) else {}),
                },
            })
    text = "".join(texts)
    message = {"role": "assistant", "content": text if text != "" else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    prompt_tokens = number_or_zero(usage.get("promptTokenCount"))
    completion_tokens = number_or_zero(usage.get("candidatesTokenCount"))
    total = usage.get("totalTokenCount")
    total_tokens = total if isinstance(total, (int, float)) else prompt_tokens + completion_tokens
    return {
        "id": "chatcmpl-agg-%d" % request_seq,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason_of_gemini(cand.get("finishReason"), bool(tool_calls))}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": total_tokens},
    }


def gemini_to_anthropic(body, model, request_seq):
    """A Gemini generateContent answer as an Anthropic Messages object."""
    openai = gemini_to_openai(body, model, request_seq)
    choices = openai["choices"]
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    content = []
    text = message.get("content")
    if isinstance(text, str) and text != "":
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls", []):
        content.append({
            "type": "tool_use",
            "id": call.get("id", "toolu_unknown"),
            "name": call["function"].get("name", ""),
            "input": arguments_of(call["function"].get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = openai["usage"]
    return {
        "id": "msg_agg-%d" % request_seq,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of_openai(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]},
    }


def anthropic_message_to_openai(body, model, request_seq):
    """An Anthropic Messages object as a Chat Completions object."""
    content = body.get("content") if isinstance(body.get("content"), list) else []
    texts = []
    tool_calls = []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
                "type": "function",
                "function": {
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                },
            })
    text = "".join(texts)
    message = {"role": "assistant", "content": text if text != "" else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    prompt_tokens = number_or_zero(usage.get("input_tokens"))
    completion_tokens = number_or_zero(usage.get("output_tokens"))
    return {
        "id": "chatcmpl-" + (body.get("id") if isinstance(body.get("id"), str) else "agg-%d" % request_seq),
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason_of_anthropic(body.get("stop_reason"))}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


# ------------------------------------------------------------------ SSE ---

def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":")))


def parse_sse(text):
    """SSE text -> list of data payload strings ('[DONE]' included as-is)."""
    out = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        frame = frame.strip("\n")
        if frame == "":
            continue
        payloads = []
        for line in frame.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                payloads.append(line[5:].lstrip())
        for p in payloads:
            out.append(p)
    return out


def openai_chunk(model, request_seq, delta=None, finish_reason=None, usage=None, chunk_id=None):
    chunk = {
        "id": chunk_id or ("chatcmpl-agg-%d" % request_seq),
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def anthropic_events_to_openai_chunks(events, model, request_seq, include_usage):
    """Anthropic SSE event payloads -> OpenAI chat chunk dicts."""
    chunks = []
    chunk_id = None
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    tool_ids = {}
    started = False

    def ensure_started():
        nonlocal started
        if not started:
            started = True
            chunks.append(openai_chunk(model, request_seq, delta={"role": "assistant"}, chunk_id=chunk_id))

    for payload in events:
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        if not is_record(ev):
            continue
        etype = ev.get("type")
        if etype == "message_start":
            msg = ev.get("message") if is_record(ev.get("message")) else {}
            if isinstance(msg.get("id"), str):
                chunk_id = "chatcmpl-" + msg["id"]
            usage = msg.get("usage") if is_record(msg.get("usage")) else {}
            input_tokens = number_or_zero(usage.get("input_tokens"))
            ensure_started()
        elif etype == "content_block_delta":
            idx = ev.get("index")
            delta = ev.get("delta") if is_record(ev.get("delta")) else {}
            ensure_started()
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str) and delta["text"] != "":
                chunks.append(openai_chunk(model, request_seq, delta={"content": delta["text"]}, chunk_id=chunk_id))
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str) and delta["partial_json"] != "":
                tid = tool_ids.get(idx, "toolu_%s" % idx)
                chunks.append(openai_chunk(model, request_seq, delta={"tool_calls": [{
                    "index": idx if isinstance(idx, int) else 0, "id": tid, "type": "function",
                    "function": {"arguments": delta["partial_json"]}}]}, chunk_id=chunk_id))
        elif etype == "content_block_start":
            block = ev.get("content_block") if is_record(ev.get("content_block")) else {}
            idx = ev.get("index")
            if block.get("type") == "tool_use":
                tool_ids[idx] = block.get("id") if isinstance(block.get("id"), str) else "toolu_%s" % idx
                ensure_started()
                chunks.append(openai_chunk(model, request_seq, delta={"tool_calls": [{
                    "index": idx if isinstance(idx, int) else 0, "id": tool_ids[idx], "type": "function",
                    "function": {"name": block.get("name") if isinstance(block.get("name"), str) else "",
                                 "arguments": ""}}]}, chunk_id=chunk_id))
        elif etype == "message_delta":
            d = ev.get("delta") if is_record(ev.get("delta")) else {}
            if isinstance(d.get("stop_reason"), str):
                stop_reason = d["stop_reason"]
            usage = ev.get("usage") if is_record(ev.get("usage")) else {}
            output_tokens = number_or_zero(usage.get("output_tokens"))
    ensure_started()
    finish = finish_reason_of_anthropic(stop_reason)
    final = openai_chunk(model, request_seq, delta={}, finish_reason=finish, chunk_id=chunk_id)
    if include_usage:
        final["usage"] = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens}
    chunks.append(final)
    return chunks


def gemini_frames_to_openai_chunks(frames, model, request_seq, include_usage):
    """Gemini :streamGenerateContent SSE payloads -> OpenAI chat chunk dicts."""
    chunks = []
    started = False
    tool_count = 0
    input_tokens = 0
    output_tokens = 0
    finish = "stop"

    def ensure_started():
        nonlocal started
        if not started:
            started = True
            chunks.append(openai_chunk(model, request_seq, delta={"role": "assistant"}))

    for payload in frames:
        try:
            frame = json.loads(payload)
        except Exception:
            continue
        if not is_record(frame):
            continue
        usage = frame.get("usageMetadata") if is_record(frame.get("usageMetadata")) else {}
        if usage:
            input_tokens = number_or_zero(usage.get("promptTokenCount"))
            output_tokens = number_or_zero(usage.get("candidatesTokenCount"))
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        cand = candidates[0] if candidates and is_record(candidates[0]) else {}
        content = cand.get("content") if is_record(cand.get("content")) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        tool_here = False
        for part in parts:
            if not is_record(part):
                continue
            if isinstance(part.get("text"), str) and part["text"] != "":
                ensure_started()
                chunks.append(openai_chunk(model, request_seq, delta={"content": part["text"]}))
            if is_record(part.get("functionCall")):
                fc = part["functionCall"]
                tool_here = True
                ensure_started()
                chunks.append(openai_chunk(model, request_seq, delta={"tool_calls": [{
                    "index": tool_count, "id": "call_gemini_%d" % tool_count, "type": "function",
                    "function": {"name": fc.get("name") if isinstance(fc.get("name"), str) else "",
                                 "arguments": json.dumps(fc.get("args") if is_record(fc.get("args")) else {})}}]}))
                tool_count += 1
        if isinstance(cand.get("finishReason"), str):
            finish = finish_reason_of_gemini(cand["finishReason"], tool_here or tool_count > 0)
    ensure_started()
    final = openai_chunk(model, request_seq, delta={}, finish_reason=finish)
    if include_usage:
        final["usage"] = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens}
    chunks.append(final)
    return chunks


def openai_chunks_to_anthropic_events(chunks, model, request_seq):
    """OpenAI chat chunk dicts -> Anthropic SSE event strings (port of the
    starter's anthropicEventsFromOpenAI, operating on parsed chunks)."""
    events = []
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0
    # The usage chunk arrives last, but message_start is emitted first, so
    # pre-scan the chunks for usage before generating any event.
    for _chunk in chunks:
        _usage = _chunk.get("usage") if is_record(_chunk) else None
        if is_record(_usage):
            input_tokens = number_or_zero(_usage.get("prompt_tokens"))
            output_tokens = number_or_zero(_usage.get("completion_tokens"))

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})]

    def close_events():
        nonlocal text_index
        out = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        return close_events() + [
            sse_event("message_delta", {"type": "message_delta",
                                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                        "usage": {"output_tokens": output_tokens}}),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    for chunk in chunks:
        if not is_record(chunk):
            continue
        usage = chunk.get("usage") if is_record(chunk.get("usage")) else {}
        if usage:
            input_tokens = number_or_zero(usage.get("prompt_tokens"))
            output_tokens = number_or_zero(usage.get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        events.extend(start_events())
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if text_index == -1:
                text_index = next_index
                next_index += 1
                events.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""}}))
            events.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": delta["content"]}}))
        calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
        for call in calls:
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                events.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use",
                                      "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else "toolu_%d" % index,
                                      "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                                      "input": {}}}))
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                events.append(sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}))
        if choice.get("finish_reason") is not None:
            pending_stop = stop_reason_of_openai(choice["finish_reason"])
    if not finished:
        events.extend(start_events())
        events.extend(finish_events(pending_stop or "end_turn"))
    return events


# ------------------------------------------------------------ server state ---

directory = {"schema": PROVIDERS_SCHEMA, "providers": []}
ratings = None
workload = None
probes = []
catalog = {"providers": []}
calls = {}       # provider id -> provider calls made so far (this trial)
refusals = {}    # provider id -> 402/429/5xx refusals so far
consecutive_refusals = {}  # provider id -> refusals in a row, reset by a success
exhausted = set()  # providers that refused twice in a row: a scheduled fail
                   # is a lone refusal, a spent budget refuses every call
manifest_index = 0
request_seq = 0


def served_models():
    """The free models across the providers that answered, in the brief's
    provider order, models sorted by id: what GET /v1/models lists."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


def candidates_for(requires, json_mode):
    """Free models meeting the requirements, ranked best first: by rating,
    then by how little their provider has been used (calls plus three per
    refusal), then by the brief's provider order, then by model id."""
    out = []
    for pi, probe in enumerate(probes):
        if not probe["ok"]:
            continue
        if json_mode and probe["provider"]["wire"] != "openai":
            # A JSON-mode request carries response_format on the OpenAI wire;
            # a provider whose wire cannot carry it echoes that it was absent.
            continue
        for model in probe["models"]:
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            rating = row["rating_milli"] if row is not None else 0
            out.append({
                "provider": probe["provider"],
                "model": model,
                "rating": rating,
                "order": (pi, model["id"]),
            })
    out.sort(key=lambda c: (-c["rating"],
                            calls.get(c["provider"]["id"], 0) + 3 * refusals.get(c["provider"]["id"], 0),
                            c["order"][0], c["order"][1]))
    # A provider that refused twice in a row has spent its budget (or hit its
    # outage): stop spending calls on it while any other candidate exists.
    fresh = [c for c in out if c["provider"]["id"] not in exhausted]
    return fresh if fresh else out


def requirements_from_body(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


def requirements_for(body):
    """The requirements of the request now arriving: the manifest's row by
    arrival order, else the body's own."""
    global manifest_index
    manifest = workload["manifest"] if workload is not None else []
    entry = manifest[manifest_index] if manifest_index < len(manifest) else None
    manifest_index += 1
    if entry is not None:
        return entry["requires"]
    return requirements_from_body(body)


# --------------------------------------------------------- upstream calls ---

def build_upstream_body(client_wire, candidate, body, stream, requires):
    """The request body for the candidate's wire, naming its model."""
    wire = candidate["provider"]["wire"]
    model = candidate["model"]["id"]
    json_mode = requires.get("json") is True
    if wire == "openai":
        if client_wire == "openai":
            out = dict(body)
            out["model"] = model
            if stream:
                # The client checks a streamed answer for its final usage, so
                # always ask the provider for a usage chunk, even when the
                # client did not send stream_options itself.
                out["stream_options"] = {"include_usage": True}
            return out
        return anthropic_to_openai(body, model, json_mode=json_mode)
    if wire == "anthropic":
        if client_wire == "anthropic":
            out = dict(body)
            out["model"] = model
            return out
        # OpenAI request -> Anthropic Messages. Only the fields the wire
        # carries are translated; the rest is preserved in the shapes below.
        return openai_to_anthropic_request(body, model, stream)
    # gemini
    if client_wire == "openai":
        return openai_to_gemini(body, model, json_mode=json_mode)
    return anthropic_to_gemini(body, model, json_mode=json_mode)


def openai_to_anthropic_request(body, model, stream):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    messages = []
    system = ""
    tools_out = []
    messages_in = body.get("messages") if isinstance(body.get("messages"), list) else []
    for msg in messages_in:
        if not is_record(msg):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system = content if isinstance(content, str) else text_of(content)
            continue
        if role == "tool":
            # A tool result becomes a user turn carrying one tool_result block.
            block = {"type": "tool_result",
                     "tool_use_id": msg.get("tool_call_id") if isinstance(msg.get("tool_call_id"), str) else "",
                     "content": content if isinstance(content, str) else json.dumps(content)}
            messages.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks = []
            text = content if isinstance(content, str) else text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls", []) if isinstance(msg.get("tool_calls"), list) else []:
                if not is_record(tc) or not is_record(tc.get("function")):
                    continue
                blocks.append({"type": "tool_use",
                               "id": tc.get("id") if isinstance(tc.get("id"), str) else "",
                               "name": tc["function"].get("name") if isinstance(tc["function"].get("name"), str) else "",
                               "input": arguments_of(tc["function"].get("arguments"))})
            messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            blocks = []
            for p in content:
                if not is_record(p):
                    continue
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    blocks.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url" and is_record(p.get("image_url")):
                    url = p["image_url"].get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        try:
                            header, b64 = url[5:].split(",", 1)
                            blocks.append({"type": "image", "source": {
                                "type": "base64", "media_type": header.split(";")[0] or "image/png",
                                "data": b64}})
                        except ValueError:
                            pass
                    elif isinstance(url, str) and url != "":
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system != "":
        out["system"] = system
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        for t in body["tools"]:
            if not is_record(t) or not is_record(t.get("function")):
                continue
            fn = t["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools_out.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools_out:
            out["tools"] = tools_out
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
    if stream:
        out["stream"] = True
    return out


def call_provider(candidate, body, stream):
    """One call to the candidate's provider. Returns (status, content_type,
    body_bytes). Raises on connection failure."""
    provider = candidate["provider"]
    base = join_base(ENV_URL, provider["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    if provider["wire"] == "openai":
        url = base + "/v1/chat/completions"
        if stream:
            headers["accept"] = "text/event-stream"
    elif provider["wire"] == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if stream:
            headers["accept"] = "text/event-stream"
    else:
        action = ":streamGenerateContent" if stream else ":generateContent"
        url = "%s/v1beta/models/%s%s" % (base, candidate["model"]["id"], action)
        if stream:
            url += "?alt=sse"
        headers["accept"] = "text/event-stream" if stream else "application/json"
    status, resp_headers, data = http_request("POST", url, headers=headers, body=payload,
                                              timeout=UPSTREAM_TIMEOUT_S)
    ctype = ""
    for k, v in resp_headers.items():
        if k.lower() == "content-type":
            ctype = v
            break
    return status, ctype, data


def normalize_upstream(candidate, status, data, stream, model, seq, include_usage):
    """An upstream answer as OpenAI shape: ('json', dict) or ('sse', [chunks])."""
    wire = candidate["provider"]["wire"]
    if wire == "openai":
        if stream:
            frames = [p for p in parse_sse(data.decode("utf-8", "replace")) if p != "[DONE]"]
            chunks = []
            for p in frames:
                try:
                    chunks.append(json.loads(p))
                except Exception:
                    continue
            return "sse", chunks
        try:
            return "json", json.loads(data.decode("utf-8"))
        except Exception:
            return "json", {}
    if wire == "anthropic":
        if stream:
            frames = parse_sse(data.decode("utf-8", "replace"))
            return "sse", anthropic_events_to_openai_chunks(frames, model, seq, include_usage)
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            return "json", {}
        return "json", anthropic_message_to_openai(msg, model, seq)
    # gemini
    if stream:
        frames = parse_sse(data.decode("utf-8", "replace"))
        return "sse", gemini_frames_to_openai_chunks(frames, model, seq, include_usage)
    try:
        gen = json.loads(data.decode("utf-8"))
    except Exception:
        return "json", {}
    return "json", gemini_to_openai(gen, model, seq)


def render_to_client(client_wire, normalized, requested_model, seq):
    """Normalized OpenAI shape -> (status, content_type, body_bytes) in the
    wire the client used."""
    kind, payload = normalized
    if kind == "json":
        if client_wire == "openai":
            return 200, "application/json", json.dumps(payload, separators=(",", ":")).encode("utf-8")
        out = openai_to_anthropic(payload, requested_model)
        return 200, "application/json", json.dumps(out, separators=(",", ":")).encode("utf-8")
    # streaming
    if client_wire == "openai":
        parts = []
        for chunk in payload:
            parts.append("data: %s\n\n" % json.dumps(chunk, separators=(",", ":")))
        parts.append("data: [DONE]\n\n")
        return 200, "text/event-stream", "".join(parts).encode("utf-8")
    events = openai_chunks_to_anthropic_events(payload, requested_model, seq)
    return 200, "text/event-stream", "".join(events).encode("utf-8")


def serve(client_wire, body):
    """Answer one client request. Returns (status, content_type, body_bytes)."""
    global request_seq
    seq = request_seq
    request_seq += 1
    requires = requirements_for(body)
    stream = body.get("stream") is True
    json_mode = requires.get("json") is True
    requested_model = body.get("model") if isinstance(body.get("model"), str) and body.get("model") != "" else None
    tried = set()
    for cand in candidates_for(requires, json_mode):
        pid = cand["provider"]["id"]
        if pid in tried:
            continue
        upstream_body = build_upstream_body(client_wire, cand, body, stream, requires)
        try:
            status, _ctype, data = call_provider(cand, upstream_body, stream)
        except Exception as e:
            log("call to %s failed: %s" % (pid, e))
            tried.add(pid)
            continue
        calls[pid] = calls.get(pid, 0) + 1
        if status == 200:
            consecutive_refusals[pid] = 0
            model = cand["model"]["id"]
            # A streamed answer must carry its final usage, so always ask
            # the normalizer for it; the client's stream_options decide
            # nothing here.
            normalized = normalize_upstream(cand, status, data, stream, model, seq, True)
            return render_to_client(client_wire, normalized,
                                    requested_model or model, seq)
        tried.add(pid)
        if status in (402, 429) or status >= 500:
            refusals[pid] = refusals.get(pid, 0) + 1
            consecutive_refusals[pid] = consecutive_refusals.get(pid, 0) + 1
            if consecutive_refusals[pid] >= 2:
                exhausted.add(pid)
                log("provider %s looks exhausted, skipping it from now on" % pid)
        log("provider %s refused %s with HTTP %d" % (pid, cand["model"]["id"], status))
    if client_wire == "anthropic":
        err = {"type": "error", "error": {"type": "overloaded_error",
                                           "message": "no free model on the directory meets this request"}}
    else:
        err = {"error": {"message": "no free model on the directory meets this request",
                         "type": "no_free_model", "code": None}}
    return 503, "application/json", json.dumps(err, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------- handler ---

def send_json(handler, status, body):
    text = json.dumps(body, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(text)))
    handler.end_headers()
    handler.wfile.write(text)


def send_error(handler, wire, status, err_type, message):
    if wire == "anthropic":
        send_json(handler, status, {"type": "error", "error": {"type": err_type, "message": message}})
    else:
        send_json(handler, status, {"error": {"message": message, "type": err_type, "code": None}})


def openai_model_list(models):
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models]}


def anthropic_model_list(models):
    return {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _send(self, status, content_type, data):
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/v1/models":
            models = served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                send_json(self, 200, anthropic_model_list(models))
            else:
                send_json(self, 200, openai_model_list(models))
            return
        send_error(self, "openai", 404, "not_found", "no route GET %s" % self.path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            wire = "openai"
        elif path == "/v1/messages":
            wire = "anthropic"
        else:
            send_error(self, "anthropic" if path == "/v1/messages" else "openai",
                       404, "not_found", "no route POST %s" % path)
            return
        body = self._read_json()
        if body is None:
            send_error(self, wire, 400, "invalid_request_error",
                       "the request body is not one JSON object")
            return
        try:
            status, content_type, data = serve(wire, body)
        except Exception as e:
            log("serve failed: %s" % e)
            send_error(self, wire, 500, "internal_error", "the server failed to answer")
            return
        self._send(status, content_type, data)


# ------------------------------------------------------------------- main ---

def register(port):
    res_status, _hdrs, data = http_request(
        "POST", ENV_URL + "/register",
        headers={"content-type": "application/json"},
        body=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                         "catalog": catalog}, separators=(",", ":")).encode("utf-8"),
        timeout=5)
    if res_status < 200 or res_status >= 300:
        raise RuntimeError("POST %s/register answered HTTP %d: %s"
                           % (ENV_URL, res_status, data[:500].decode("utf-8", "replace")))
    try:
        reply = json.loads(data.decode("utf-8"))
    except Exception:
        reply = {}
    if is_record(reply) and isinstance(reply.get("requests"), (int, float)):
        log("registered: the workload holds %d requests" % reply["requests"])


def poll_work(server):
    while True:
        threading.Event().wait(POLL_INTERVAL_S)
        try:
            status, _hdrs, data = http_request("GET", ENV_URL + "/work", timeout=2)
            if status == 200:
                work = json.loads(data.decode("utf-8"))
                if is_record(work) and work.get("done") is True:
                    server.shutdown()
                    return
        except Exception:
            pass


def main():
    global directory, ratings, workload, probes, catalog
    directory, ratings, workload, err = read_brief(PROMPT_FILE)
    if err is not None:
        log(err)
    if ENV_URL != "":
        probes = [probe_provider(ENV_URL, p) for p in directory["providers"]]
        for probe in probes:
            if not probe["ok"] and probe["error"] is not None:
                log(probe["error"])
    catalog = build_catalog(probes, ratings)
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        sys.exit(1)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        sys.exit(1)
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]), len(served_models())))
    register(port)
    threading.Thread(target=poll_work, args=(server,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(str(e))
        sys.exit(1)
