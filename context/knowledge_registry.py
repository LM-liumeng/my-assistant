"""Persistent registry for multiple local knowledge bases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeRegistry:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.registry_root = self.base_dir / "rag"
        self.registry_path = self.registry_root / "knowledge_bases.json"
        self.default_input_root = self.base_dir / "knowledge"
        self.default_output_root = self.base_dir / "rag"

    def list_bases(self) -> List[Dict[str, Any]]:
        data = self._load()
        active_id = data.get("active_knowledge_base", "default")
        rows: List[Dict[str, Any]] = []
        for kb_id, item in sorted((data.get("knowledge_bases") or {}).items()):
            row = dict(item or {})
            row["id"] = kb_id
            row["is_active"] = kb_id == active_id
            rows.append(row)
        return rows

    def get_base(self, knowledge_base: Optional[str]) -> Dict[str, Any]:
        kb_id = self.normalize_id(knowledge_base or self.get_active_id())
        data = self._load()
        kb = (data.get("knowledge_bases") or {}).get(kb_id)
        if kb:
            row = dict(kb)
            row["id"] = kb_id
            row["is_active"] = kb_id == data.get("active_knowledge_base")
            return row
        return self.upsert_base(kb_id, name=knowledge_base or kb_id)

    def get_active_id(self) -> str:
        return self._load().get("active_knowledge_base", "default")

    def set_active(self, knowledge_base: str) -> Dict[str, Any]:
        kb = self.get_base(knowledge_base)
        data = self._load()
        data["active_knowledge_base"] = kb["id"]
        self._save(data)
        kb["is_active"] = True
        return kb

    def upsert_base(
        self,
        knowledge_base: str,
        name: Optional[str] = None,
        input_dir: Optional[str] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        kb_id = self.normalize_id(knowledge_base)
        data = self._load()
        knowledge_bases = data.setdefault("knowledge_bases", {})
        existing = dict(knowledge_bases.get(kb_id) or {})
        resolved_input = self._resolve_input_dir(input_dir, kb_id)
        resolved_output = self._resolve_output_dir(kb_id)
        updated = {
            "id": kb_id,
            "name": (name or existing.get("name") or kb_id).strip(),
            "description": description.strip() or existing.get("description", ""),
            "input_dir": str(resolved_input),
            "output_dir": str(resolved_output),
        }
        knowledge_bases[kb_id] = updated
        if not data.get("active_knowledge_base"):
            data["active_knowledge_base"] = kb_id
        self._save(data)
        updated["is_active"] = kb_id == data.get("active_knowledge_base")
        return updated

    def resolve_input_dir(self, knowledge_base: Optional[str]) -> Path:
        return Path(self.get_base(knowledge_base)["input_dir"])

    def resolve_output_dir(self, knowledge_base: Optional[str]) -> Path:
        return Path(self.get_base(knowledge_base)["output_dir"])

    def _resolve_output_dir(self, kb_id: str) -> Path:
        legacy_output = self.default_output_root.resolve()
        if kb_id == 'default':
            has_legacy_index = any((legacy_output / name).exists() for name in ['ingestion_manifest.json', 'chunks.jsonl', 'embeddings.npy'])
            if has_legacy_index:
                return legacy_output
        return (self.default_output_root / kb_id).resolve()

    def normalize_id(self, value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "default"
        lowered = text.lower()
        lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "-", lowered)
        lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
        return lowered or "default"

    def _resolve_input_dir(self, input_dir: Optional[str], kb_id: str) -> Path:
        raw = (input_dir or "").strip()
        if raw:
            path = Path(raw).expanduser()
            return path if path.is_absolute() else (self.base_dir / path).resolve()
        default_dir = self.default_input_root / kb_id
        if kb_id == "default":
            legacy_dir = self.default_input_root
            if legacy_dir.exists():
                return legacy_dir.resolve()
        return default_dir.resolve()

    def _load(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            data = self._default_data()
            self._save(data)
            return data
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            data = self._default_data()
            self._save(data)
            return data

    def _save(self, data: Dict[str, Any]) -> None:
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _default_data(self) -> Dict[str, Any]:
        input_dir = self.default_input_root.resolve()
        output_dir = self._resolve_output_dir('default')
        return {
            "active_knowledge_base": "default",
            "knowledge_bases": {
                "default": {
                    "id": "default",
                    "name": "默认知识库",
                    "description": "项目默认知识库",
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                }
            },
        }
