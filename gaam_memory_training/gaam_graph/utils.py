from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    raw = "||".join(str(p) for p in parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{h}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def canonicalize_name(name: str) -> str:
    name = normalize_text(name).strip().lower()
    name = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", name)
    return name.strip("_") or "unknown"


def read_json(path: str | Path) -> Any:
    p = Path(path)
    data = p.read_bytes()
    if orjson:
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if orjson:
        p.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    else:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
