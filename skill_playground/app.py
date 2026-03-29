"""Flask entry for the isolated skill playground."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from skill_playground.agent.orchestrator import PlaygroundOrchestrator
from skill_playground.agent.trace import TraceRecorder


PROJECT_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
orchestrator = PlaygroundOrchestrator(str(PROJECT_ROOT))
_job_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "")
    history = payload.get("history") or []
    job_id = uuid.uuid4().hex
    trace = TraceRecorder()
    trace.log("system", "Accepted new playground chat job.", {"job_id": job_id})

    with _job_lock:
        _jobs[job_id] = {
            "status": "queued",
            "trace": trace,
            "result": None,
            "error": None,
        }

    worker = threading.Thread(target=_run_chat_job, args=(job_id, message, history), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/chat/jobs/<job_id>")
def api_chat_job(job_id: str):
    with _job_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404

    trace = job.get("trace")
    trace_payload = trace.snapshot() if hasattr(trace, "snapshot") else []
    payload = {
        "job_id": job_id,
        "status": job.get("status") or "unknown",
        "trace": trace_payload,
        "result": job.get("result"),
        "error": job.get("error"),
    }
    return jsonify(payload)


@app.post("/api/skills/reload")
def api_reload_skills():
    orchestrator.refresh_skills()
    return jsonify({"message": "Skills reloaded."})


def _run_chat_job(job_id: str, message: str, history: Any) -> None:
    with _job_lock:
        job = _jobs.get(job_id)
    if job is None:
        return

    trace = job["trace"]
    trace.log("system", "Worker started.")
    _update_job(job_id, status="running")
    try:
        result = orchestrator.chat(message=message, history=history, trace=trace)
        _update_job(job_id, status="completed", result=result)
        trace.log("system", "Job finished successfully.")
    except Exception as exc:
        trace.log("system", "Job failed.", {"error": str(exc)})
        _update_job(job_id, status="failed", error=str(exc))


def _update_job(job_id: str, **updates: Any) -> None:
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(updates)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=True)
