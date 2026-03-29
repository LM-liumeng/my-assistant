"""ModelTool: runs a simple sentiment analysis model on text."""

from __future__ import annotations

from typing import Any, Dict

from context.evidence_store import EvidenceStore


class ModelTool:
    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store
        self.positive_words = {"good", "happy", "great", "excellent", "love", "fantastic", "amazing"}
        self.negative_words = {"bad", "sad", "terrible", "awful", "hate", "horrible", "poor"}

    def run(self, text: str = "", **kwargs: Any) -> Dict[str, Any]:
        normalized = (text or "").strip()
        self.evidence_store.log_event({"event": "model_run", "text": normalized})
        if not normalized:
            return {
                "error": "No text was provided for analysis.",
                "message": "Please provide the text you want analyzed.",
            }

        words = normalized.lower().split()
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        score = pos_count - neg_count

        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        result_message = f"Sentiment: {label} (score {score})"
        return {
            "message": result_message,
            "label": label,
            "score": score,
            "display_content": result_message,
            "display_filename": "model_output.txt",
        }
