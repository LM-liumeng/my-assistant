# AI Desktop Office Assistant Demo

This project demonstrates a modular AI-driven office assistant with a simple web interface. It is intended as a learning example; some capabilities such as web search, email sending, and external model calls require additional configuration before they can be used in a real environment.

## Major Components

- **Agent orchestration layer**: interprets user commands, plans actions, routes them to tools, and integrates outputs.
  - `IntentRecognizer`: recognises file search, email, document, web search, media analysis, knowledge-base, and model-analysis intents with simple heuristics in English and Chinese.
  - `Planner`: maps intents to tool calls.
  - `ToolRouter`: invokes the appropriate tool.
  - `ResultIntegrator`: merges tool outputs into a reply.

- **Context and knowledge layer**: stores metadata, supports retrieval, and logs evidence.
  - `MetadataStore` persists assistant state to `metadata.json`.
  - `RetrievalStore` indexes files in the `workspace` subdirectory under the data directory.
  - `EvidenceStore` writes audit logs to `evidence.log` and ignores IO errors so they do not interrupt the main workflow.
  - `HybridRAGStore` and related tools provide local knowledge-base retrieval.

- **Tools and execution layer**: encapsulates assistant capabilities.
  - `FileSearchTool`: searches for files in the workspace by name.
  - `DocumentTool`: creates, reads, and appends to text files in the workspace.
  - `EmailTool`: drafts and sends emails. If SMTP credentials are configured, it attempts real delivery; otherwise it logs the email locally.
  - `WebSearchTool`: performs web search through Google Custom Search when configured.
  - `ModelTool`: runs a trivial local sentiment-analysis model.
  - `VideoAnalysisTool`: performs image and video object-detection analysis.
  - `RAGTool` / `RAGManagementTool`: query and manage local knowledge bases.
  - `ChatTool`: provides open-ended conversation through an OpenAI-compatible API.

- **Safety and governance layer**: restricts side effects and requires confirmation for higher-risk operations.
  - The assistant only writes to directories registered via `SafetyLayer.add_allowed_dir`.
  - Actions with side effects, such as writing files or sending emails, can require confirmation.
  - Confirmation events are logged in `evidence.log`.

## Setup and Running

1. **Install dependencies**

   ```bash
   pip install flask
   ```

2. **Prepare the data directory**

   The assistant stores metadata, logs, documents, and RAG assets under the `data` directory. When run through `app/main.py`, the directory is created automatically if it does not already exist.

3. **Configure optional credentials**

   - **Email sending**: set `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, and `SMTP_PASSWORD` to enable real email delivery.
   - **Web search**: set `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` or `GOOGLE_CX` to enable Google Custom Search.
   - **Chat model**: set `MY_DEEPSEEK_KEY` or `DEEPSEEK_API_KEY` to enable chat completions. You can also set `DEEPSEEK_API_BASE`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_BASE_URL`, `LLM_MODEL`, and `LLM_SYSTEM_PROMPT` as needed.
   - **Confirmation behaviour**: set `AUTO_CONFIRM=1` if you want the assistant to perform actions without prompting for confirmation.

4. **Run the server**

   ```bash
   python app/main.py
   ```

   Then open `http://127.0.0.1:5000/` in your browser.

## Example Commands

- **File search**
  - English: `search for budget report`
  - Chinese: `搜索 报告` or `查找 报告`

- **Create or append documents**
  - English: `create document notes.txt with content Meeting notes...`
  - Chinese: `创建文档 notes.txt 内容 会议记录`

- **Draft and send emails**
  - English: `send email to bob@example.com subject Project Update body The project is on track`
  - Chinese: `邮件 收件人 bob@example.com 主题 会议 正文 内容`

- **Web search**
  - English: `google cats`
  - Chinese: `搜索网页 猫`

- **Run model / sentiment analysis**
  - English: `analyze sentiment: I love this movie`
  - Chinese: `分析 情感 我喜欢这部电影`

- **Media analysis**
  - Chinese: `分析 video.mp4 文件`
  - Chinese: `识别图片 demo.png`

- **Knowledge-base management**
  - Chinese: `查看知识库状态`
  - Chinese: `增量入库 D:\CodeProject\assistant_demo\data\knowledge`
  - Chinese: `重建知识库索引 D:\CodeProject\assistant_demo\data\knowledge`

## Limitations and Extension Points

- Some capabilities require external credentials or local model/runtime dependencies.
- Intent recognition is still heuristic-first and may misinterpret unconventional phrasing.
- The current confirmation model is lightweight; production deployments should use explicit UI approval flows and session-aware state.
- File retrieval is currently simple and can be extended with richer indexing strategies.
- The skill system supports prompt skills and handler skills, and now also supports rule-based recall plus semantic judgment over recalled candidates.

This project is intended as a practical foundation that can be extended with more tools, stricter policy controls, richer RAG pipelines, and stronger skill orchestration.
