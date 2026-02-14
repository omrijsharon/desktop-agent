from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class ModelCaps:
    unsupported_params: set[str]


class ModelCapsStore:
    """Persist per-model parameter compatibility.

    Goal: avoid 400s by omitting known-unsupported request parameters per model.
    """

    def __init__(self, *, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._caps: dict[str, ModelCaps] = {}
        self._loaded = False

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            for model, rec in data.items():
                if not isinstance(model, str) or not isinstance(rec, dict):
                    continue
                up = rec.get("unsupported_params")
                if isinstance(up, list):
                    self._caps[model] = ModelCaps(unsupported_params={str(x) for x in up if str(x).strip()})
        except Exception:
            return

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data: JsonDict = {}
            for model, caps in sorted(self._caps.items(), key=lambda kv: kv[0]):
                data[model] = {"unsupported_params": sorted(caps.unsupported_params)}
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            pass

    def mark_unsupported(self, *, model: str, param: str) -> None:
        model = str(model or "").strip()
        param = str(param or "").strip()
        if not model or not param:
            return
        with self._lock:
            self._load_if_needed()
            caps = self._caps.get(model)
            if caps is None:
                caps = ModelCaps(unsupported_params=set())
                self._caps[model] = caps
            if param in caps.unsupported_params:
                return
            caps.unsupported_params.add(param)
            self._save()

    def filter_create_kwargs(self, *, model: str, create_kwargs: JsonDict) -> JsonDict:
        model = str(model or "").strip()
        if not model:
            return dict(create_kwargs)
        with self._lock:
            self._load_if_needed()
            caps = self._caps.get(model)
            if caps is None or not caps.unsupported_params:
                return dict(create_kwargs)
            out = dict(create_kwargs)
            for p in list(caps.unsupported_params):
                out.pop(p, None)
            return out

    def snapshot(self) -> JsonDict:
        with self._lock:
            self._load_if_needed()
            return {m: {"unsupported_params": sorted(c.unsupported_params)} for m, c in self._caps.items()}
