"""Media answer synthesis for image/video analysis results."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent.skill_manager import SkillManager
from tools.chat_tool import ChatTool


_MEDIA_SYSTEM_PROMPT = """
You are a media-analysis assistant. Answer only from the supplied detection context.
Follow these rules:
1. Do not invent objects, actions, times, or counts.
2. When detections are absent, state that no objects were detected.
3. Prefer concise factual answers. If you infer uncertainty, say it is based on detections.
4. If the user asks what appears in the media, summarize the main detected classes and counts.
5. If the user asks for scene understanding or message inference, reason only from the supplied detections and clearly mark it as an inference.
""".strip()

_CRITIC_SYSTEM_PROMPT = """
You are reviewing a media-analysis answer. Check that it only uses the provided detection context,
that counts are consistent, and that it does not invent extra objects or events.
Return only the corrected final answer.
""".strip()


class MediaAnswerAgent:
    """Turn raw detection outputs into a more useful user-facing answer."""

    def __init__(self, chat_tool: ChatTool, skill_manager: Optional[SkillManager] = None) -> None:
        self.chat_tool = chat_tool
        self.skill_manager = skill_manager

    def answer(self, question: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        local_answer = self._build_local_answer(question, analysis_result)
        context = self._build_context(analysis_result)

        local_answer = self._postprocess(local_answer, user_query=question)

        if not self.chat_tool.is_configured():
            return {
                "message": local_answer,
                "display_content": context,
                "display_filename": analysis_result.get("display_filename", "media_analysis_summary.txt"),
            }

        draft_result = self.chat_tool.complete(
            user_prompt=f"Question:\n{question}\n\nDetection context:\n{context}\n\nBase answer:\n{local_answer}",
            system_prompt=self._augment_prompt(_MEDIA_SYSTEM_PROMPT, scope="media", user_query=question),
        )
        draft_text = draft_result.get("message", "").strip()
        if draft_result.get("error") or not draft_text:
            return {
                "message": local_answer,
                "display_content": context,
                "display_filename": analysis_result.get("display_filename", "media_analysis_summary.txt"),
            }

        critic_result = self.chat_tool.complete(
            user_prompt=f"Question:\n{question}\n\nDetection context:\n{context}\n\nDraft answer:\n{draft_text}",
            system_prompt=self._augment_prompt(_CRITIC_SYSTEM_PROMPT, scope="media", user_query=question),
        )
        final_text = critic_result.get("message", "").strip() or draft_text
        if critic_result.get("error"):
            final_text = draft_text
        final_text = self._postprocess(final_text, user_query=question)

        return {
            "message": final_text,
            "display_content": context,
            "display_filename": analysis_result.get("display_filename", "media_analysis_summary.txt"),
        }


    def _augment_prompt(self, base_prompt: str, scope: str, user_query: str = "") -> str:
        if self.skill_manager is None:
            return base_prompt
        return self.skill_manager.augment_prompt(base_prompt, scope=scope, user_query=user_query)

    def _postprocess(self, text: str, user_query: str = "") -> str:
        if self.skill_manager is None:
            return (text or "").strip()
        return self.skill_manager.postprocess_response(text, scope="media", user_query=user_query)

    def can_answer_from_context(self, question: str, display_content: Optional[str], display_filename: Optional[str]) -> bool:
        if not display_content:
            return False
        if display_filename and display_filename == "media_analysis_summary.txt":
            return True
        content = display_content.strip()
        return content.startswith("Source:") and "Detected objects:" in content

    def answer_from_context(self, question: str, display_content: str, display_filename: Optional[str]) -> Dict[str, Any]:
        parsed = self.parse_context(display_content, display_filename)
        if parsed is None:
            return {
                "error": "Unable to parse media analysis context.",
                "message": "I could not reuse the current media-analysis context.",
            }
        return self.answer(question, parsed)

    def parse_context(self, display_content: str, display_filename: Optional[str]) -> Optional[Dict[str, Any]]:
        content = (display_content or "").strip()
        if not content:
            return None
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        source = ""
        media_type = "media"
        units = 0
        unit_label = "unit"
        object_counts: Dict[str, int] = {}
        in_detections = False
        for line in lines:
            if line.startswith("Source:"):
                source = line.split(":", 1)[1].strip()
            elif line.startswith("Media type:"):
                media_type = line.split(":", 1)[1].strip() or "media"
            elif line.startswith("Analyzed "):
                match = re.search(r"Analyzed\s+(\d+)\s+(\w+)", line)
                if match:
                    units = int(match.group(1))
                    unit_label = match.group(2)
            elif line == "Detected objects:" or line == "Detections:":
                in_detections = True
            elif line == "Objects: none" or line == "Detections: none":
                in_detections = False
            elif in_detections and line.startswith("-"):
                match = re.match(r"-\s+(.+?):\s+(\d+)", line)
                if match:
                    object_counts[match.group(1).strip()] = int(match.group(2))
        return {
            "source_path": source,
            "media_type": media_type,
            "object_counts": object_counts,
            "units": units,
            "unit_label": unit_label,
            "display_filename": display_filename or "media_analysis_summary.txt",
        }

    def _build_local_answer(self, question: str, analysis_result: Dict[str, Any]) -> str:
        media_type = analysis_result.get("media_type", "media")
        source = analysis_result.get("source_path") or analysis_result.get("source") or "the selected media"
        counts: Dict[str, int] = analysis_result.get("object_counts") or {}
        units = int(analysis_result.get("units") or 0)
        unit_label = analysis_result.get("unit_label") or ("frame" if media_type == "video" else "image")
        lower_question = (question or "").lower()

        if not counts:
            if media_type == "video":
                return f"在已分析的 {units} 个{unit_label}中，未检测到明确目标。"
            return f"在图片 {source} 中，未检测到明确目标。"

        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top_items = ordered[:4]
        parts = [f"{name} {count} 个" for name, count in top_items]
        joined = "，".join(parts)

        if any(keyword in lower_question for keyword in ["推断", "infer", "内容", "message", "场景", "发生了什么", "表达了什么"]):
            return (
                f"基于当前检测结果，可以做有限推断：该{media_type}里主要出现了 {joined}。"
                f"这说明画面重点大概率围绕这些目标展开，但无法仅凭目标检测准确还原视频具体传达的信息。"
            )

        if media_type == "video":
            total_classes = len(counts)
            return (
                f"视频 {source} 的分析结果显示，在已处理的 {units} 个{unit_label}中，"
                f"主要检测到 {joined}。共检测到 {total_classes} 类目标。"
            )
        return f"图片 {source} 中主要检测到 {joined}。"

    def _build_context(self, analysis_result: Dict[str, Any]) -> str:
        lines: List[str] = [
            f"Source: {analysis_result.get('source_path') or analysis_result.get('source') or ''}",
            f"Media type: {analysis_result.get('media_type', 'media')}",
            f"Analyzed {analysis_result.get('units', 0)} {analysis_result.get('unit_label', '')}".strip(),
        ]
        counts: Dict[str, int] = analysis_result.get("object_counts") or {}
        if not counts:
            lines.append("Detections: none")
            return "\n".join(lines)
        lines.append("Detected objects:")
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {name}: {count}")
        return "\n".join(lines)
