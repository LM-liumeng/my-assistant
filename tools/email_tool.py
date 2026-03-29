"""EmailTool: drafts, sends, receives, and summarizes email messages."""

from __future__ import annotations

import imaplib
import os
import smtplib
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default
from email.utils import parseaddr
from typing import Any, Dict, List, Optional, Tuple

from context.evidence_store import EvidenceStore
from security import SafetyLayer


class EmailTool:
    def __init__(self, evidence_store: EvidenceStore, safety: SafetyLayer) -> None:
        self.evidence_store = evidence_store
        self.safety = safety

    def handle_email(self, to: str = "", subject: str = "", body: str = "") -> Dict[str, Any]:
        params = {"to": to, "subject": subject, "body": body}
        self.safety.log_tool_call("email", params)

        normalized_to, to_error = self._normalize_email_address(to)
        if to_error:
            return {"error": to_error}

        display_text = f"To: {normalized_to}\nSubject: {subject or '(no subject)'}\n\n{body or ''}"
        if not self.safety.confirm_action(f"Send email to {normalized_to} with subject '{subject}'?"):
            return {
                "message": f"Confirmation required to send email to {normalized_to}. Set AUTO_CONFIRM or confirm manually.",
                "display_content": display_text,
                "display_filename": "draft_email",
            }

        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = int(os.environ.get("SMTP_PORT", "0")) if os.environ.get("SMTP_PORT") else None
        smtp_user = os.environ.get("SMTP_USERNAME")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        if smtp_server and smtp_user and smtp_password and smtp_port:
            normalized_from, from_error = self._normalize_email_address(smtp_user, field_name="SMTP username")
            if from_error:
                return {
                    "error": from_error,
                    "display_content": display_text,
                    "display_filename": "draft_email",
                }

            try:
                msg = EmailMessage()
                msg["From"] = normalized_from
                msg["To"] = normalized_to
                msg["Subject"] = subject or "(no subject)"
                msg.set_content(body or "")
                with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
                    smtp.login(smtp_user, smtp_password)
                    smtp.send_message(msg, from_addr=normalized_from, to_addrs=[normalized_to])
                self.evidence_store.log_event(
                    {
                        "event": "email_sent",
                        "to": normalized_to,
                        "subject": subject,
                        "body": body,
                        "method": "smtp",
                    }
                )
                return {
                    "message": f"Email successfully sent to {normalized_to}.",
                    "display_content": display_text,
                    "display_filename": "draft_email",
                }
            except Exception as exc:
                self.evidence_store.log_event({"event": "email_send_failed", "error": str(exc)})
                return {
                    "error": f"Failed to send email via SMTP: {exc}",
                    "display_content": display_text,
                    "display_filename": "draft_email",
                }

        try:
            self.evidence_store.log_event(
                {
                    "event": "email_logged",
                    "to": normalized_to,
                    "subject": subject,
                    "body": body,
                    "method": "log_only",
                }
            )
        except Exception:
            pass
        return {
            "message": (
                "Email parameters recorded locally. To enable sending, set SMTP_SERVER, "
                "SMTP_PORT, SMTP_USERNAME and SMTP_PASSWORD environment variables."
            ),
            "display_content": display_text,
            "display_filename": "draft_email",
        }

    def _normalize_email_address(self, value: str, field_name: str = "Email address") -> Tuple[Optional[str], Optional[str]]:
        raw_value = (value or "").strip()
        if not raw_value:
            return None, f"{field_name} ('to') is required." if field_name == "Email address" else f"{field_name} is required."

        _name, address = parseaddr(raw_value)
        normalized = address or raw_value
        normalized = normalized.strip().strip("<>")
        if "@" not in normalized:
            return None, f"{field_name} '{raw_value}' is not a valid email address."

        local_part, domain = normalized.rsplit("@", 1)
        if not local_part or not domain:
            return None, f"{field_name} '{raw_value}' is not a valid email address."

        try:
            local_part.encode("ascii")
        except UnicodeEncodeError:
            return None, (
                f"{field_name} '{raw_value}' uses a non-ASCII mailbox name. "
                "The current SMTP flow requires an ASCII local part unless the server supports SMTPUTF8."
            )

        try:
            ascii_domain = domain.encode("idna").decode("ascii")
        except UnicodeError:
            return None, f"{field_name} '{raw_value}' has an invalid domain name."

        return f"{local_part}@{ascii_domain}", None

    def handle_receive(self, folder: str = "INBOX", num_emails: int = 5, search_criteria: str = "ALL") -> List[Dict[str, Any]]:
        imap_server = os.environ.get("IMAP_SERVER")
        imap_port = int(os.environ.get("IMAP_PORT", "993"))
        imap_user = os.environ.get("IMAP_USERNAME")
        imap_password = os.environ.get("IMAP_PASSWORD")

        if not all([imap_server, imap_user, imap_password]):
            return [{"error": "IMAP credentials not configured. Set IMAP_SERVER, IMAP_PORT, IMAP_USERNAME, IMAP_PASSWORD."}]

        try:
            mail = imaplib.IMAP4_SSL(imap_server, imap_port)
            mail.login(imap_user, imap_password)
            mail.select(folder)

            status, messages = mail.search(None, search_criteria)
            if status != "OK":
                raise Exception("Search failed.")

            email_ids = messages[0].split()[-num_emails:]
            emails = []

            for email_id in email_ids:
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = msg_data[0][1]
                msg = BytesParser(policy=default).parsebytes(raw_email)
                emails.append(
                    {
                        "from": msg["From"],
                        "subject": msg["Subject"],
                        "body": self._get_email_body(msg),
                    }
                )

            mail.close()
            mail.logout()
            self.evidence_store.log_event({"event": "emails_received", "count": len(emails), "folder": folder})
            return emails
        except Exception as exc:
            self.evidence_store.log_event({"event": "imap_receive_failed", "error": str(exc)})
            return [{"error": f"Failed to receive emails: {exc}"}]

    def _get_email_body(self, msg: EmailMessage) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    return payload.decode() if payload else ""
        else:
            payload = msg.get_payload(decode=True)
            return payload.decode() if payload else ""
        return ""

    def handle_summarize(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summaries = []
        for email in emails:
            body = email.get("body", "")
            summary = body[:100] + "..." if len(body) > 100 else body
            summaries.append(
                {
                    "from": email.get("from"),
                    "subject": email.get("subject"),
                    "summary": summary,
                }
            )

        self.evidence_store.log_event({"event": "emails_summarized", "count": len(summaries)})
        return summaries
