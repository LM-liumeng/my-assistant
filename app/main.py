"""Flask web application exposing the desktop assistant."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from agent.orchestrator import Agent


def _restore_auto_confirm(previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop("AUTO_CONFIRM", None)
    else:
        os.environ["AUTO_CONFIRM"] = previous_value


def create_app() -> Flask:
    app = Flask(__name__)
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    base_dir = os.path.join(project_dir, "data")
    os.makedirs(base_dir, exist_ok=True)
    agent = Agent(base_dir)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api/command", methods=["POST"])
    def api_command():
        data = request.get_json(force=True) or {}
        command = data.get("command", "").strip()
        display_content = data.get("display_content")
        display_filename = data.get("display_filename")
        knowledge_base = data.get("knowledge_base", "").strip() or None
        if not command:
            return jsonify({"message": "Please enter a command."})
        try:
            response = agent.handle(
                command,
                display_content=display_content,
                display_filename=display_filename,
                knowledge_base=knowledge_base,
            )
            return jsonify(response)
        except Exception as exc:
            msg = f"Internal server error: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/confirm_email", methods=["POST"])
    def api_confirm_email():
        data = request.get_json(force=True) or {}
        to_addr = data.get("to", "").strip()
        subject = data.get("subject", "").strip()
        body = data.get("body", "")
        if not to_addr:
            return jsonify({"error": "Email address ('to') is required."}), 400

        previous_auto_confirm = os.environ.get("AUTO_CONFIRM")
        os.environ["AUTO_CONFIRM"] = "1"
        try:
            result = agent.email_tool.handle_email(to=to_addr, subject=subject, body=body)
        finally:
            _restore_auto_confirm(previous_auto_confirm)
        return jsonify(result)

    @app.route("/api/confirm_document", methods=["POST"])
    def api_confirm_document():
        data = request.get_json(force=True) or {}
        filename = data.get("filename", "").strip()
        content = data.get("content", "")
        if not filename:
            return jsonify({"error": "Filename must be specified."}), 400

        previous_auto_confirm = os.environ.get("AUTO_CONFIRM")
        os.environ["AUTO_CONFIRM"] = "1"
        try:
            result = agent.document_tool.handle_document(filename=filename, content=content)
        finally:
            _restore_auto_confirm(previous_auto_confirm)
        return jsonify(result)

    @app.route("/api/file/<path:filename>", methods=["GET", "POST"])
    def api_file(filename: str):
        workspace = os.path.join(base_dir, "workspace")
        safe_path = os.path.abspath(os.path.join(workspace, filename))
        if not safe_path.startswith(os.path.abspath(workspace)):
            return jsonify({"error": "Access denied."}), 403

        if request.method == "GET":
            if not os.path.exists(safe_path):
                return jsonify({"error": f"File '{filename}' not found."}), 404
            try:
                with open(safe_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                return jsonify({"content": content})
            except Exception as exc:
                return jsonify({"error": f"Failed to read file: {exc}"}), 500

        data = request.get_json(force=True) or {}
        content = data.get("content", "")
        result = agent.document_tool.handle_document(filename=filename, content=content)
        status_code = 400 if "error" in result else 200
        return jsonify(result), status_code

    @app.route("/api/display/clear", methods=["POST"])
    def api_clear_display():
        try:
            return jsonify(agent.clear_display_context())
        except Exception as exc:
            msg = f"Failed to clear display context: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/kbs", methods=["GET"])
    def api_rag_kbs():
        try:
            return jsonify(agent.rag_management_tool.list_knowledge_bases())
        except Exception as exc:
            msg = f"Failed to load knowledge-base registry: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/tools", methods=["GET"])
    def api_tools():
        try:
            return jsonify(agent.list_registered_tools())
        except Exception as exc:
            msg = f"Failed to load tool registry: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/capabilities", methods=["GET"])
    def api_capabilities():
        try:
            return jsonify(agent.list_capabilities())
        except Exception as exc:
            msg = f"Failed to load capability registry: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/mcp/refresh", methods=["POST"])
    def api_mcp_refresh():
        try:
            result = agent.refresh_dynamic_tools()
            status_code = 400 if "error" in result else 200
            return jsonify(result), status_code
        except Exception as exc:
            msg = f"Failed to refresh MCP tools: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/kbs", methods=["POST"])
    def api_rag_save_kb():
        data = request.get_json(force=True) or {}
        knowledge_base = data.get("knowledge_base", "").strip()
        knowledge_base_name = data.get("knowledge_base_name", "").strip()
        input_dir = data.get("input_dir", "").strip()
        description = data.get("description", "").strip()
        try:
            result = agent.rag_management_tool.save_knowledge_base(
                knowledge_base=knowledge_base,
                knowledge_base_name=knowledge_base_name,
                input_dir=input_dir,
                description=description,
            )
            status_code = 400 if "error" in result else 200
            return jsonify(result), status_code
        except Exception as exc:
            msg = f"Failed to save knowledge-base config: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/select", methods=["POST"])
    def api_rag_select():
        data = request.get_json(force=True) or {}
        knowledge_base = data.get("knowledge_base", "").strip()
        try:
            result = agent.rag_management_tool.select_knowledge_base(knowledge_base)
            status_code = 400 if "error" in result else 200
            return jsonify(result), status_code
        except Exception as exc:
            msg = f"Failed to select knowledge base: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/status", methods=["GET"])
    def api_rag_status():
        knowledge_base = request.args.get("knowledge_base", "").strip()
        try:
            return jsonify(agent.rag_management_tool.status(knowledge_base=knowledge_base))
        except Exception as exc:
            msg = f"Failed to load RAG status: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/contents", methods=["GET"])
    def api_rag_contents():
        knowledge_base = request.args.get("knowledge_base", "").strip()
        limit = int(request.args.get("limit", 12))
        offset = int(request.args.get("offset", 0))
        try:
            return jsonify(agent.rag_management_tool.browse_contents(knowledge_base=knowledge_base, limit=limit, offset=offset))
        except Exception as exc:
            msg = f"Failed to load RAG contents: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/ingest", methods=["POST"])
    def api_rag_ingest():
        data = request.get_json(force=True) or {}
        knowledge_base = data.get("knowledge_base", "").strip()
        input_dir = data.get("input_dir", "").strip()
        rebuild = bool(data.get("rebuild", False))
        run_async = bool(data.get("async", False))
        try:
            if run_async:
                result = agent.rag_management_tool.start_ingestion(
                    knowledge_base=knowledge_base,
                    input_dir=input_dir,
                    rebuild=rebuild,
                )
                status_code = 400 if "error" in result else 202
                return jsonify(result), status_code
            result = agent.rag_management_tool.run_ingestion(
                knowledge_base=knowledge_base,
                input_dir=input_dir,
                rebuild=rebuild,
            )
            status_code = 400 if "error" in result else 200
            return jsonify(result), status_code
        except Exception as exc:
            msg = f"Failed to run RAG ingestion: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/rag/jobs/<job_id>", methods=["GET"])
    def api_rag_job(job_id: str):
        try:
            result = agent.rag_management_tool.get_job(job_id)
            status_code = 404 if "error" in result else 200
            return jsonify(result), status_code
        except Exception as exc:
            msg = f"Failed to load RAG job: {exc}"
            return jsonify({"error": msg, "message": msg}), 500

    @app.route("/api/receive_emails", methods=["POST"])
    def api_receive_emails():
        data = request.get_json(force=True) or {}
        folder = data.get("folder", "INBOX").strip()
        num_emails = int(data.get("num_emails", 5))
        search_criteria = data.get("search_criteria", "ALL").strip()

        previous_auto_confirm = os.environ.get("AUTO_CONFIRM")
        os.environ["AUTO_CONFIRM"] = "1"
        try:
            emails = agent.email_tool.handle_receive(folder=folder, num_emails=num_emails, search_criteria=search_criteria)
            return jsonify({"emails": emails})
        except Exception as exc:
            return jsonify({"error": f"Failed to receive emails: {exc}"}), 500
        finally:
            _restore_auto_confirm(previous_auto_confirm)

    @app.route("/api/summarize_emails", methods=["POST"])
    def api_summarize_emails():
        data = request.get_json(force=True) or {}
        emails = data.get("emails", [])
        if not emails or not isinstance(emails, list):
            return jsonify({"error": "A list of emails is required."}), 400

        previous_auto_confirm = os.environ.get("AUTO_CONFIRM")
        os.environ["AUTO_CONFIRM"] = "1"
        try:
            summaries = agent.email_tool.handle_summarize(emails)
            return jsonify({"summaries": summaries})
        except Exception as exc:
            return jsonify({"error": f"Failed to summarize emails: {exc}"}), 500
        finally:
            _restore_auto_confirm(previous_auto_confirm)

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug_flag = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug_flag)
