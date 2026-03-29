import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.orchestrator import Agent, IntentRecognizer
from agent.media_answerer import MediaAnswerAgent
from context.evidence_store import EvidenceStore
from security import SafetyLayer
from tools.chat_tool import ChatTool
from tools.email_tool import EmailTool
from tools.video_tool import VideoAnalysisTool


class DummyChatTool(ChatTool):
    def __init__(self) -> None:
        self.evidence_store = EvidenceStore('.')

    def is_configured(self) -> bool:
        return False


def test_intent_recognizes_media_path_as_media():
    recognizer = IntentRecognizer(intent_tool=None)
    assert recognizer.is_media_path(r"D:\demo\clip.mp4") is True
    assert recognizer.is_media_path("note.txt") is False


def test_intent_recognizes_clear_display_command():
    recognizer = IntentRecognizer(intent_tool=None)
    intent, params = recognizer.recognise("\u6e05\u7a7a\u80cc\u666f\u4fe1\u606f\u7a97\u53e3")
    assert intent == "clear_display"
    assert params == {}


def test_generic_analyze_uses_display_text_for_run_model(tmp_path: Path):
    agent = Agent(str(tmp_path))
    response = agent.handle("分析", display_content="I love this product")
    assert response["message"] == "Sentiment: positive (score 1)"


def test_generic_analyze_uses_media_file_when_display_filename_is_media(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"not-a-real-image")
    agent = Agent(str(tmp_path))
    response = agent.handle("分析", display_filename=str(image_path))
    assert "ultralytics" in response["message"] or "Failed to open image" in response["message"] or "does not exist" in response["message"]


def test_intent_recognizes_media_analysis_for_quoted_path():
    recognizer = IntentRecognizer(intent_tool=None)
    command = 'analyze image "D:\\Media Files\\demo clip.mp4" and detect objects'
    intent, params = recognizer.recognise(command)
    assert intent == "media_analysis"
    assert params["source"] == r"D:\Media Files\demo clip.mp4"


def test_media_tool_resolves_existing_quoted_image_path(tmp_path: Path):
    image_path = tmp_path / "demo image.png"
    image_path.write_bytes(b"test")
    tool = VideoAnalysisTool(EvidenceStore(str(tmp_path)), SafetyLayer(EvidenceStore(str(tmp_path))), base_dir=str(tmp_path))
    resolved, media_type = tool._resolve_source(f'"{image_path}"')
    assert resolved == str(image_path)
    assert media_type == "image"


def test_media_answer_agent_builds_smarter_local_summary():
    answerer = MediaAnswerAgent(DummyChatTool())
    result = answerer.answer(
        "video.mp4中出现了哪些目标",
        {
            "source_path": "video.mp4",
            "media_type": "video",
            "object_counts": {"person": 5, "car": 2},
            "units": 30,
            "unit_label": "frame",
            "display_filename": "media_analysis_summary.txt",
        },
    )
    assert "视频 video.mp4 的分析结果显示" in result["message"]
    assert "person 5 个" in result["message"]
    assert "car 2 个" in result["message"]


def test_media_answer_agent_can_reuse_display_context():
    answerer = MediaAnswerAgent(DummyChatTool())
    assert answerer.can_answer_from_context(
        "能否根据已检测到的内容推断视频内容发送了什么",
        "Source: video.mp4\nMedia type: video\nAnalyzed 30 frame\nDetected objects:\n- person: 5\n- car: 2",
        "media_analysis_summary.txt",
    ) is True


def test_agent_uses_smart_media_answer_for_detected_objects(tmp_path: Path):
    agent = Agent(str(tmp_path))
    agent.video_tool.analyze = lambda **kwargs: {
        "message": "Video analysis completed. Detected 2 object class(es).",
        "display_content": "Source: video.mp4\nDetected objects:\n- person: 5\n- car: 2",
        "display_filename": "media_analysis_summary.txt",
        "source_path": "video.mp4",
        "media_type": "video",
        "object_counts": {"person": 5, "car": 2},
        "units": 30,
        "unit_label": "frame",
    }
    response = agent.handle("video.mp4中出现了哪些目标")
    assert "视频 video.mp4 的分析结果显示" in response["message"]
    assert response["display_filename"] == "media_analysis_summary.txt"


def test_agent_can_answer_followup_from_existing_media_context(tmp_path: Path):
    agent = Agent(str(tmp_path))
    response = agent.handle(
        "能否根据已检测到的内容推断视频内容发送了什么",
        display_content="Source: video.mp4\nMedia type: video\nAnalyzed 30 frame\nDetected objects:\n- person: 5\n- car: 2",
        display_filename="media_analysis_summary.txt",
    )
    assert "基于当前检测结果，可以做有限推断" in response["message"]
    assert "person 5 个" in response["message"]


def test_agent_remembers_media_context_for_followup_without_display_payload(tmp_path: Path):
    agent = Agent(str(tmp_path))
    agent.video_tool.analyze = lambda **kwargs: {
        "message": "Video analysis completed. Detected 2 object class(es).",
        "display_content": "Source: video.mp4\nMedia type: video\nAnalyzed 30 frame\nDetected objects:\n- person: 5\n- car: 2",
        "display_filename": "media_analysis_summary.txt",
        "source_path": "video.mp4",
        "media_type": "video",
        "object_counts": {"person": 5, "car": 2},
        "units": 30,
        "unit_label": "frame",
    }
    agent.handle("video.mp4中出现了哪些目标")
    response = agent.handle("能否根据已检测到的内容推断视频内容发送了什么")
    assert "基于当前检测结果，可以做有限推断" in response["message"]
    assert "person 5 个" in response["message"]


def test_chinese_email_intent_extracts_all_fields():
    recognizer = IntentRecognizer(intent_tool=None)
    intent, params = recognizer.recognise("邮件 收件人 bob@example.com 主题 测试邮件 正文 你好世界")
    assert intent == "email"
    assert params == {"to": "bob@example.com", "subject": "测试邮件", "body": "你好世界"}


def test_english_email_intent_extracts_address_without_subject_tail():
    recognizer = IntentRecognizer(intent_tool=None)
    intent, params = recognizer.recognise("send email to bob@example.com subject Test update body Hello there")
    assert intent == "email"
    assert params["to"] == "bob@example.com"
    assert params["subject"] == "Test update"
    assert params["body"] == "Hello there"


def test_email_tool_normalizes_idn_domain_and_keeps_ascii_local_part(tmp_path: Path):
    tool = EmailTool(EvidenceStore(str(tmp_path)), SafetyLayer(EvidenceStore(str(tmp_path))))
    normalized, error = tool._normalize_email_address("user@例子.测试")
    assert error is None
    assert normalized == "user@xn--fsqu00a.xn--0zwm56d"


def test_email_tool_rejects_non_ascii_mailbox_name(tmp_path: Path):
    tool = EmailTool(EvidenceStore(str(tmp_path)), SafetyLayer(EvidenceStore(str(tmp_path))))
    normalized, error = tool._normalize_email_address("测试@exampl.com")
    assert normalized is None
    assert "non-ASCII mailbox name" in error


def test_chat_response_is_exposed_as_display_content_for_followup_save(tmp_path: Path):
    agent = Agent(str(tmp_path))

    class StubChatTool:
        def is_configured(self) -> bool:
            return True

        def complete_messages(self, messages, system_prompt=None):
            return {"message": "stub answer"}

    agent.chat_tool = StubChatTool()
    response = agent.handle("hello there")
    assert response["display_content"] == "stub answer"

    save_response = agent.handle("save as answer.txt")
    assert save_response["display_content"] == "stub answer"
    assert save_response["display_filename"] == "answer.txt"


def test_clear_display_command_clears_active_context(tmp_path: Path):
    agent = Agent(str(tmp_path))

    class StubChatTool:
        def is_configured(self) -> bool:
            return True

        def complete_messages(self, messages, system_prompt=None):
            return {"message": "stub answer"}

    agent.chat_tool = StubChatTool()
    response = agent.handle("hello there")
    assert response["display_content"] == "stub answer"
    assert agent.memory.active_artifact is not None
    assert agent.memory.last_assistant_message == "stub answer"

    cleared = agent.handle("\u6e05\u7a7a\u80cc\u666f\u4fe1\u606f\u7a97\u53e3")
    assert cleared["message"] == "\u5df2\u6e05\u7a7a\u80cc\u666f\u4fe1\u606f\u7a97\u53e3\u3002"
    assert cleared["display_content"] == ""
    assert cleared["display_filename"] == ""
    assert agent.memory.active_artifact is None
    assert agent.memory.last_assistant_message is None


def test_email_request_uses_colloquial_chinese_body_extraction(tmp_path: Path):
    agent = Agent(str(tmp_path))
    response = agent.handle("\u53d1\u9001\u90ae\u4ef6\u7ed91907957364@qq.com\uff0c\u544a\u8bc9\u4ed6\u6211\u5728\u6d4b\u8bd5\u7a0b\u5e8f")
    assert "Confirmation required to send email" in response["message"]
    assert "To: 1907957364@qq.com" in response["display_content"]
    assert "\u6211\u5728\u6d4b\u8bd5\u7a0b\u5e8f" in response["display_content"]


def test_chat_tool_supports_common_deepseek_env_names(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MY_DEEPSEEK_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    from tools.chat_tool import ChatTool
    tool = ChatTool(EvidenceStore(str(tmp_path)))
    assert tool.is_configured() is True
    assert tool.api_key == "test-key"


def test_intent_llm_web_search_parse_is_ignored_without_explicit_search_prefix(tmp_path: Path):
    class StubIntentTool:
        def parse_intent(self, command):
            return {"intent": "web_search", "args": {"query": command}}

    recognizer = IntentRecognizer(intent_tool=StubIntentTool())
    intent, params = recognizer.recognise("Tell me about this assistant")
    assert intent == "chat"
    assert params["prompt"] == "Tell me about this assistant"
