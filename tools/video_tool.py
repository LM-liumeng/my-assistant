"""Media analysis tool using YOLO object detection."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import cv2  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

from context.evidence_store import EvidenceStore
from security import SafetyLayer


class VideoAnalysisTool:
    """Perform object detection on images and videos and summarise the results."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm", ".mpeg", ".mpg"}

    def __init__(self, evidence_store: EvidenceStore, safety: SafetyLayer, base_dir: str | None = None) -> None:
        self.evidence_store = evidence_store
        self.safety = safety
        self.workspace_dir: str | None = None
        if base_dir:
            abs_base = os.path.abspath(base_dir)
            self.workspace_dir = os.path.join(abs_base, "workspace")
            self.project_dir = abs_base
        else:
            self.project_dir = None
        self._model_cache: Dict[str, Any] = {}

    def analyze(self, source: str = "", weights: str | None = None, max_frames: int = 100) -> Dict[str, Any]:
        params = {"source": source, "weights": weights, "max_frames": max_frames}
        self.safety.log_tool_call("media_analysis", params)
        if not source:
            return {"error": "A media path or webcam index is required."}

        resolved_source, media_type = self._resolve_source(source)
        if resolved_source is None or media_type is None:
            return {"error": f"Media file '{source}' does not exist or is not a supported image/video path."}

        if YOLO is None:
            return {
                "error": (
                    "YOLO inference is unavailable because the 'ultralytics' package is not installed. "
                    "Install 'ultralytics' to analyze images and videos."
                )
            }

        weights_path = self._resolve_weights_path(weights)
        if weights_path not in self._model_cache:
            try:
                model = YOLO(weights_path)  # type: ignore[operator]
                self._model_cache[weights_path] = model
            except Exception as exc:
                return {"error": f"Failed to load YOLO model from '{weights_path}': {exc}"}
        else:
            model = self._model_cache[weights_path]

        if media_type == "image":
            result = self._analyze_image(model, resolved_source)
        else:
            result = self._analyze_video(model, resolved_source, max_frames=max_frames)
        if "error" in result:
            return result

        object_counts = result["object_counts"]
        units = result["units"]
        unit_label = result["unit_label"]
        summary = self._format_summary(
            source_path=resolved_source,
            media_type=media_type,
            object_counts=object_counts,
            units=units,
            unit_label=unit_label,
        )
        if object_counts:
            message = f"{media_type.capitalize()} analysis completed. Detected {len(object_counts)} object class(es)."
        else:
            message = f"{media_type.capitalize()} analysis completed. No objects were detected."
        payload = {
            "message": message,
            "display_content": summary,
            "display_filename": "media_analysis_summary.txt",
            "source_path": str(resolved_source),
            "media_type": media_type,
            "object_counts": object_counts,
            "units": units,
            "unit_label": unit_label,
        }
        try:
            self.evidence_store.log_event({"event": "video_analysis_completed", **payload})
        except Exception:
            pass
        return payload

    def _strip_wrapping_quotes(self, value: str) -> str:
        text = (value or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            return text[1:-1].strip()
        return text

    def _candidate_paths(self, source: str) -> Iterable[str]:
        cleaned = self._strip_wrapping_quotes(source)
        if not cleaned:
            return []
        candidates: list[str] = []
        if os.path.isabs(cleaned):
            candidates.append(cleaned)
        else:
            candidates.append(os.path.abspath(cleaned))
            if self.workspace_dir:
                candidates.append(os.path.abspath(os.path.join(self.workspace_dir, cleaned)))
            if self.project_dir:
                candidates.append(os.path.abspath(os.path.join(self.project_dir, cleaned)))
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def _resolve_source(self, source: str) -> tuple[Any | None, str | None]:
        cleaned = self._strip_wrapping_quotes(source)
        if cleaned.isdigit():
            return int(cleaned), "video"
        for candidate in self._candidate_paths(cleaned):
            if os.path.exists(candidate):
                return candidate, self._infer_media_type(candidate)
        return None, None

    def _infer_media_type(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in self.IMAGE_EXTENSIONS:
            return "image"
        if ext in self.VIDEO_EXTENSIONS:
            return "video"
        image = cv2.imread(path)
        if image is not None:
            return "image"
        return "video"

    def _resolve_weights_path(self, weights: str | None) -> str:
        configured = self._strip_wrapping_quotes(weights or os.environ.get("YOLO_WEIGHTS", "")).strip()
        candidates: list[str] = []
        if configured:
            if os.path.isabs(configured):
                candidates.append(configured)
            else:
                candidates.extend(self._candidate_paths(configured))
        local_default = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
        candidates.extend([local_default, "yolov8n.pt"])
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return configured or local_default

    def _collect_object_counts(self, results: Any) -> Dict[str, int]:
        object_counts: Dict[str, int] = {}
        if not results:
            return object_counts
        for result in results:
            try:
                boxes = result.boxes
                names = result.names
                for cls_index in boxes.cls:
                    cls_name = names.get(int(cls_index), str(int(cls_index)))
                    object_counts[cls_name] = object_counts.get(cls_name, 0) + 1
            except Exception:
                continue
        return object_counts

    def _merge_counts(self, total: Dict[str, int], partial: Dict[str, int]) -> None:
        for key, value in partial.items():
            total[key] = total.get(key, 0) + value

    def _analyze_image(self, model: Any, source_path: str) -> Dict[str, Any]:
        frame = cv2.imread(source_path)
        if frame is None:
            return {"error": f"Failed to open image '{source_path}'."}
        try:
            results = model.predict(frame, verbose=False)
        except Exception as exc:
            return {"error": f"YOLO prediction failed: {exc}"}
        return {"object_counts": self._collect_object_counts(results), "units": 1, "unit_label": "image"}

    def _analyze_video(self, model: Any, source_path: Any, max_frames: int) -> Dict[str, Any]:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video source '{source_path}'."}
        object_counts: Dict[str, int] = {}
        frames_processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_processed += 1
            try:
                results = model.predict(frame, verbose=False)
            except Exception as exc:
                cap.release()
                return {"error": f"YOLO prediction failed: {exc}"}
            self._merge_counts(object_counts, self._collect_object_counts(results))
            if max_frames > 0 and frames_processed >= max_frames:
                break
        cap.release()
        return {"object_counts": object_counts, "units": frames_processed, "unit_label": "frame"}

    def _format_summary(
        self,
        source_path: Any,
        media_type: str,
        object_counts: Dict[str, int],
        units: int,
        unit_label: str,
    ) -> str:
        lines = [
            f"Source: {source_path}",
            f"Media type: {media_type}",
            f"Analyzed {units} {unit_label}{'' if units == 1 else 's'}",
        ]
        if not object_counts:
            lines.append("Objects: none")
            return "\n".join(lines)
        lines.append("Detected objects:")
        for name, count in sorted(object_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {name}: {count}")
        return "\n".join(lines)
