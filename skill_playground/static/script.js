const chatLog = document.getElementById("chat-log");
const traceLog = document.getElementById("trace-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const statusText = document.getElementById("status-text");
const traceStatus = document.getElementById("trace-status");
const sendBtn = document.getElementById("send-btn");
const reloadSkillsBtn = document.getElementById("reload-skills-btn");
const messageTemplate = document.getElementById("message-template");
const traceTemplate = document.getElementById("trace-template");

const history = [];
let activeJobId = null;

function appendMessage(role, content, meta = "", skills = "") {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(`message--${role}`);
  node.querySelector(".message-meta").textContent = meta;
  node.querySelector(".message-body").textContent = content;
  node.querySelector(".message-skills").textContent = skills;
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderTrace(entries = []) {
  traceLog.innerHTML = "";
  if (!entries.length) {
    const empty = document.createElement("article");
    empty.className = "trace-entry";
    empty.innerHTML = '<div class="trace-entry__message">当前还没有调用日志。</div>';
    traceLog.appendChild(empty);
    return;
  }

  for (const entry of entries) {
    const node = traceTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".trace-entry__meta").textContent = `#${entry.index} · ${entry.timestamp} · ${entry.stage}`;
    node.querySelector(".trace-entry__message").textContent = entry.message || "";
    const detailsNode = node.querySelector(".trace-entry__details");
    const details = entry.details && Object.keys(entry.details).length
      ? JSON.stringify(entry.details, null, 2)
      : "";
    detailsNode.textContent = details;
    detailsNode.hidden = !details;
    traceLog.appendChild(node);
  }
  traceLog.scrollTop = traceLog.scrollHeight;
}

function setBusy(isBusy, text) {
  sendBtn.disabled = isBusy;
  reloadSkillsBtn.disabled = isBusy;
  statusText.textContent = text;
}

async function pollJob(jobId) {
  while (activeJobId === jobId) {
    const response = await fetch(`/api/chat/jobs/${jobId}`);
    const data = await response.json();

    renderTrace(data.trace || []);
    traceStatus.textContent = data.status || "unknown";
    const lastEntry = Array.isArray(data.trace) && data.trace.length ? data.trace[data.trace.length - 1] : null;
    if (lastEntry) {
      statusText.textContent = `${data.status || "处理中"} · ${lastEntry.message}`;
    }

    if (data.status === "completed") {
      const result = data.result || {};
      const meta = result.primary_skill ? `助手 · primary skill: ${result.primary_skill}` : "助手";
      const skills = Array.isArray(result.active_skills) && result.active_skills.length
        ? `生效 skills: ${result.active_skills.join(" / ")}`
        : "未命中动态 skill，使用基础对话链路。";
      appendMessage("assistant", result.message || "未收到回复。", meta, skills);
      history.push({ role: "assistant", content: result.message || "" });
      setBusy(false, result.error ? `已完成，但带错误信息: ${result.error}` : "完成");
      activeJobId = null;
      return;
    }

    if (data.status === "failed") {
      appendMessage("assistant", `请求失败：${data.error || "Unknown error"}`, "助手", "");
      history.push({ role: "assistant", content: `请求失败：${data.error || "Unknown error"}` });
      setBusy(false, "请求失败");
      activeJobId = null;
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 350));
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message || activeJobId) {
    return;
  }

  appendMessage("user", message, "你", "");
  history.push({ role: "user", content: message });
  chatInput.value = "";
  renderTrace([]);
  traceStatus.textContent = "queued";
  setBusy(true, "正在启动 skill 调用任务...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    const data = await response.json();
    activeJobId = data.job_id;
    traceStatus.textContent = data.status || "queued";
    await pollJob(activeJobId);
  } catch (error) {
    appendMessage("assistant", `请求失败：${error}`, "助手", "");
    history.push({ role: "assistant", content: `请求失败：${error}` });
    setBusy(false, "请求失败");
    traceStatus.textContent = "failed";
    activeJobId = null;
  }
}

async function reloadSkills() {
  if (activeJobId) {
    return;
  }
  setBusy(true, "正在重载 skills...");
  try {
    const response = await fetch("/api/skills/reload", { method: "POST" });
    const data = await response.json();
    appendMessage("assistant", data.message || "Skills reloaded.", "系统", "");
    history.push({ role: "assistant", content: data.message || "Skills reloaded." });
    renderTrace([
      {
        index: 1,
        timestamp: new Date().toLocaleTimeString(),
        stage: "system",
        message: data.message || "Skills reloaded.",
        details: {},
      },
    ]);
    traceStatus.textContent = "idle";
    setBusy(false, "skills 已重载");
  } catch (error) {
    appendMessage("assistant", `重载失败：${error}`, "系统", "");
    history.push({ role: "assistant", content: `重载失败：${error}` });
    setBusy(false, "重载失败");
    traceStatus.textContent = "failed";
  }
}

chatForm.addEventListener("submit", sendMessage);
reloadSkillsBtn.addEventListener("click", reloadSkills);
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

appendMessage(
  "assistant",
  "这是独立的 skill playground。这里不会调用主应用的文件、邮件、RAG 或媒体能力，只测试 skill 路由与会话输出。",
  "系统",
  "当前只保留 chat + skill 相关链路。"
);
renderTrace([
  {
    index: 1,
    timestamp: new Date().toLocaleTimeString(),
    stage: "system",
    message: "等待新的会话请求。发送消息后，右侧将实时刷新 skill 调用日志。",
    details: {},
  },
]);
