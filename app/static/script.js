// Client-side logic for the AI assistant web interface

document.addEventListener('DOMContentLoaded', function () {
  const chatHistory = document.getElementById('chat-history');
  const userInput = document.getElementById('user-input');
  const sendButton = document.getElementById('send-button');
  const displayFilename = document.getElementById('display-filename');
  const loadFileBtn = document.getElementById('load-file');
  const saveFileBtn = document.getElementById('save-file');
  const clearDisplayBtn = document.getElementById('clear-display');
  const displayContent = document.getElementById('display-content');

  const ragKbSelect = document.getElementById('rag-kb-select');
  const ragKbId = document.getElementById('rag-kb-id');
  const ragKbName = document.getElementById('rag-kb-name');
  const ragInputDir = document.getElementById('rag-input-dir');
  const ragKbDescription = document.getElementById('rag-kb-description');
  const ragRefreshBtn = document.getElementById('rag-refresh-btn');
  const ragSelectBtn = document.getElementById('rag-select-btn');
  const ragSaveBtn = document.getElementById('rag-save-btn');
  const ragStatusBtn = document.getElementById('rag-status-btn');
  const ragIngestBtn = document.getElementById('rag-ingest-btn');
  const ragRebuildBtn = document.getElementById('rag-rebuild-btn');
  const ragContentsBtn = document.getElementById('rag-contents-btn');
  const ragPanelStatus = document.getElementById('rag-panel-status');

  let pendingEmailText = null;
  let pendingDocument = null;
  let ragRegistry = [];
  let activeKnowledgeBase = '';
  let ragJobPollTimer = null;

  function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.textContent = text;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function showJsonParseError(text) {
    appendMessage('assistant', 'Server returned a non-JSON response: ' + text.slice(0, 200));
  }

  function updateDisplayState(data) {
    if (data.display_content !== undefined && data.display_content !== null) {
      displayContent.value = data.display_content;
    }
    if (data.display_filename !== undefined && data.display_filename !== null) {
      displayFilename.value = data.display_filename;
    }
  }

  function setRagPanelStatus(text, isError = false) {
    if (!ragPanelStatus) return;
    ragPanelStatus.textContent = text;
    ragPanelStatus.style.color = isError ? '#c04b5a' : '';
  }

  function parseEmailDetails(displayText) {
    const lines = (displayText || '').split('\n');
    let to = '';
    let subject = '';
    let body = '';

    if (lines.length > 0 && lines[0].toLowerCase().startsWith('to:')) {
      to = lines[0].substring(3).trim();
    }
    if (lines.length > 1 && lines[1].toLowerCase().startsWith('subject:')) {
      subject = lines[1].substring(8).trim();
    }

    const bodyStart = lines.findIndex((line, idx) => idx > 1 && line.trim() === '');
    if (bodyStart !== -1) {
      body = lines.slice(bodyStart + 1).join('\n');
    } else {
      body = lines.slice(2).join('\n');
    }

    return { to, subject, body };
  }

  function removeConfirmButton(id) {
    const btn = document.getElementById(id);
    if (btn) {
      btn.remove();
    }
  }

  function resetPendingActions() {
    pendingEmailText = null;
    pendingDocument = null;
    removeConfirmButton('confirm-send-btn');
    removeConfirmButton('confirm-write-btn');
  }

  function syncKnowledgeBaseState(data) {
    if (Array.isArray(data.knowledge_bases)) {
      ragRegistry = data.knowledge_bases;
      renderKnowledgeBases();
    }
    if (data.active_knowledge_base) {
      activeKnowledgeBase = data.active_knowledge_base;
      if (ragKbSelect) {
        ragKbSelect.value = activeKnowledgeBase;
      }
      fillKnowledgeBaseForm(findKnowledgeBase(activeKnowledgeBase));
    } else if (data.knowledge_base && data.knowledge_base.id) {
      activeKnowledgeBase = activeKnowledgeBase || data.knowledge_base.id;
      fillKnowledgeBaseForm(data.knowledge_base);
    }
  }

  function confirmSendEmail() {
    if (!pendingEmailText) return;

    fetch('/api/confirm_email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parseEmailDetails(pendingEmailText))
    })
      .then((res) => res.text())
      .then((text) => {
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          showJsonParseError(text);
          return;
        }

        appendMessage('assistant', data.message || (data.error ? 'Error: ' + data.error : ''));
        updateDisplayState(data);
      })
      .catch((err) => {
        appendMessage('assistant', 'Error: ' + err);
      });

    pendingEmailText = null;
    removeConfirmButton('confirm-send-btn');
  }

  function confirmWriteDocument() {
    if (!pendingDocument) return;

    fetch('/api/confirm_document', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: pendingDocument.filename,
        content: pendingDocument.content
      })
    })
      .then((res) => res.text())
      .then((text) => {
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          showJsonParseError(text);
          return;
        }

        appendMessage('assistant', data.message || (data.error ? 'Error: ' + data.error : ''));
        updateDisplayState(data);
      })
      .catch((err) => {
        appendMessage('assistant', 'Error: ' + err);
      });

    pendingDocument = null;
    removeConfirmButton('confirm-write-btn');
  }

  function saveFileWithExplicitConfirmation(filename, content) {
    const confirmed = window.confirm('确认将当前展示区内容保存到 ' + filename + ' 吗？');
    if (!confirmed) {
      appendMessage('assistant', '已取消保存。');
      return;
    }

    fetch('/api/confirm_document', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, content })
    })
      .then((res) => res.text())
      .then((text) => {
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          showJsonParseError(text);
          return;
        }

        const msg = data.message || (data.error ? 'Save failed: ' + data.error : 'File saved.');
        appendMessage('assistant', msg);
        updateDisplayState(data);
      })
      .catch((err) => {
        appendMessage('assistant', 'Save failed: ' + err.message);
      });
  }

  function maybeShowConfirmation(data) {
    if (!(data.message && data.display_content !== undefined && data.display_content !== null)) return;

    if (data.message.startsWith('Confirmation required to send email')) {
      pendingEmailText = data.display_content;
      removeConfirmButton('confirm-send-btn');
      const confirmBtn = document.createElement('button');
      confirmBtn.id = 'confirm-send-btn';
      confirmBtn.textContent = 'Confirm Send';
      confirmBtn.className = 'confirm-button';
      confirmBtn.addEventListener('click', confirmSendEmail);
      chatHistory.appendChild(confirmBtn);
      chatHistory.scrollTop = chatHistory.scrollHeight;
      return;
    }

    if (data.message.startsWith('Confirmation required to write to') && data.display_filename) {
      pendingDocument = {
        filename: data.display_filename,
        content: data.display_content
      };
      removeConfirmButton('confirm-write-btn');
      const confirmBtn = document.createElement('button');
      confirmBtn.id = 'confirm-write-btn';
      confirmBtn.textContent = 'Confirm Write';
      confirmBtn.className = 'confirm-button';
      confirmBtn.addEventListener('click', confirmWriteDocument);
      chatHistory.appendChild(confirmBtn);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  }

  function handleStructuredResponse(data, defaultErrorPrefix = 'Error', silent = false) {
    const msg = data.message || (data.error ? defaultErrorPrefix + ': ' + data.error : '');
    if (msg && !silent) {
      appendMessage('assistant', msg);
    }
    updateDisplayState(data);
    if (data.display_content === '' && data.display_filename === '') {
      resetPendingActions();
    }
    syncKnowledgeBaseState(data);
    maybeShowConfirmation(data);
    return data;
  }

  function handleCommandResponse(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      showJsonParseError(text);
      return;
    }
    handleStructuredResponse(data);
  }

  function getCurrentKnowledgeBase() {
    return (ragKbSelect && ragKbSelect.value) || (ragKbId && ragKbId.value.trim()) || activeKnowledgeBase || '';
  }

  function sendCommand() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';

    fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        command: text,
        display_content: displayContent.value || null,
        display_filename: displayFilename.value || null,
        knowledge_base: getCurrentKnowledgeBase() || null,
      })
    })
      .then((res) => res.text())
      .then(handleCommandResponse)
      .catch((err) => {
        appendMessage('assistant', 'Error: ' + err);
      });
  }

  function loadFile() {
    const filename = displayFilename.value.trim();
    if (!filename) {
      alert('Please enter a filename to load.');
      return;
    }

    fetch('/api/file/' + encodeURIComponent(filename))
      .then((res) => res.text().then((text) => ({ ok: res.ok, text })))
      .then(({ ok, text }) => {
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          showJsonParseError(text);
          return;
        }

        if (!ok) {
          appendMessage('assistant', 'Load failed: ' + (data.error || 'File not found.'));
          return;
        }

        displayContent.value = data.content || '';
        appendMessage('assistant', 'Loaded file: ' + filename);
      })
      .catch((err) => {
        appendMessage('assistant', 'Load failed: ' + err.message);
      });
  }

  function saveFile() {
    const filename = displayFilename.value.trim();
    if (!filename) {
      alert('Please enter a filename to save.');
      return;
    }

    const content = displayContent.value || '';
    fetch('/api/file/' + encodeURIComponent(filename), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    })
      .then((res) => res.text())
      .then((text) => {
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          showJsonParseError(text);
          return;
        }

        if (data.message && data.message.startsWith('Confirmation required to write to')) {
          saveFileWithExplicitConfirmation(filename, content);
          return;
        }

        const msg = data.message || (data.error ? 'Save failed: ' + data.error : 'File saved.');
        appendMessage('assistant', msg);
        updateDisplayState(data);
      })
      .catch((err) => {
        appendMessage('assistant', 'Save failed: ' + err.message);
      });
  }

  function clearDisplayWindow() {
    requestJson('/api/display/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }, '清空')
      .then(() => {
        resetPendingActions();
      })
      .catch((err) => {
        appendMessage('assistant', '清空失败: ' + err.message);
      });
  }

  function requestJson(endpoint, options, defaultErrorPrefix, silent = false) {
    return fetch(endpoint, options)
      .then((res) => res.text().then((text) => ({ ok: res.ok, text })))
      .then(({ ok, text }) => {
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          showJsonParseError(text);
          throw new Error('响应解析失败');
        }
        handleStructuredResponse(data, defaultErrorPrefix, silent);
        if (!ok || data.error) {
          throw new Error(data.error || defaultErrorPrefix + ' failed');
        }
        return data;
      });
  }

  function stopRagJobPolling() {
    if (ragJobPollTimer) {
      clearTimeout(ragJobPollTimer);
      ragJobPollTimer = null;
    }
  }

  function pollRagJob(jobId) {
    stopRagJobPolling();
    requestJson('/api/rag/jobs/' + encodeURIComponent(jobId), { method: 'GET' }, 'RAG', true)
      .then((data) => {
        const status = data.status || '';
        if (status === 'queued' || status === 'running') {
          setRagPanelStatus(data.message || '\u540e\u53f0\u4efb\u52a1\u5904\u7406\u4e2d...');
          ragJobPollTimer = window.setTimeout(function () {
            pollRagJob(jobId);
          }, 1200);
          return;
        }
        if (data.result) {
          handleStructuredResponse(data.result);
        }
        if (status === 'completed') {
          setRagPanelStatus(data.message || '\u540e\u53f0\u4efb\u52a1\u5df2\u5b8c\u6210');
          loadKnowledgeBases();
          return;
        }
        setRagPanelStatus(data.message || '\u540e\u53f0\u4efb\u52a1\u6267\u884c\u5931\u8d25', true);
      })
      .catch((err) => {
        setRagPanelStatus(err.message || '\u83b7\u53d6\u4efb\u52a1\u72b6\u6001\u5931\u8d25', true);
      });
  }

  function startRagIngestion(rebuild) {
    const payload = buildKbPayload();
    if (!payload.knowledge_base) {
      setRagPanelStatus('\u8bf7\u8f93\u5165\u77e5\u8bc6\u5e93\u7f16\u53f7', true);
      return;
    }
    setRagPanelStatus(rebuild ? '\u6b63\u5728\u91cd\u5efa\u77e5\u8bc6\u5e93\u7d22\u5f15...' : '\u6b63\u5728\u6267\u884c\u589e\u91cf\u5165\u5e93...');
    requestJson('/api/rag/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        knowledge_base: payload.knowledge_base,
        input_dir: payload.input_dir,
        rebuild: rebuild,
        async: true
      })
    }, 'RAG', true)
      .then((data) => {
        if (!data.job_id) {
          throw new Error('\u670d\u52a1\u7aef\u6ca1\u6709\u8fd4\u56de\u4efb\u52a1\u7f16\u53f7');
        }
        setRagPanelStatus(data.message || '\u540e\u53f0\u4efb\u52a1\u5df2\u5b8c\u6210');
        pollRagJob(data.job_id);
      })
      .catch((err) => {
        setRagPanelStatus(err.message, true);
      });
  }

  function requestRag(endpoint, options, successStatusText) {
    setRagPanelStatus('处理中...');
    requestJson(endpoint, options, 'RAG')
      .then((data) => {
        setRagPanelStatus(successStatusText || (data.message || '完成'));
      })
      .catch((err) => {
        setRagPanelStatus(err.message, true);
      });
  }

  function findKnowledgeBase(id) {
    const target = (id || '').trim();
    return ragRegistry.find((item) => item.id === target) || null;
  }

  function fillKnowledgeBaseForm(item) {
    const kb = item || findKnowledgeBase(getCurrentKnowledgeBase());
    if (!kb) {
      return;
    }
    if (ragKbId) ragKbId.value = kb.id || '';
    if (ragKbName) ragKbName.value = kb.name || '';
    if (ragInputDir) ragInputDir.value = kb.input_dir || '';
    if (ragKbDescription) ragKbDescription.value = kb.description || '';
  }

  function renderKnowledgeBases() {
    if (!ragKbSelect) return;
    const previous = getCurrentKnowledgeBase();
    ragKbSelect.innerHTML = '';
    ragRegistry.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.is_active ? item.name + ' (' + item.id + ', 当前)' : item.name + ' (' + item.id + ')';
      ragKbSelect.appendChild(option);
    });
    const nextValue = activeKnowledgeBase || previous || (ragRegistry[0] && ragRegistry[0].id) || '';
    ragKbSelect.value = nextValue;
    fillKnowledgeBaseForm(findKnowledgeBase(nextValue));
  }

  function loadKnowledgeBases() {
    setRagPanelStatus('加载知识库列表...');
    requestJson('/api/rag/kbs', { method: 'GET' }, 'RAG', true)
      .then((data) => {
        setRagPanelStatus('知识库列表已刷新');
        syncKnowledgeBaseState(data);
      })
      .catch((err) => {
        setRagPanelStatus(err.message, true);
      });
  }

  function buildKbPayload() {
    return {
      knowledge_base: (ragKbId && ragKbId.value.trim()) || getCurrentKnowledgeBase(),
      knowledge_base_name: (ragKbName && ragKbName.value.trim()) || '',
      input_dir: (ragInputDir && ragInputDir.value.trim()) || '',
      description: (ragKbDescription && ragKbDescription.value.trim()) || ''
    };
  }

  function bindRagPanel() {
    if (!ragKbSelect) {
      return;
    }

    ragKbSelect.addEventListener('change', function () {
      const selected = findKnowledgeBase(ragKbSelect.value);
      activeKnowledgeBase = ragKbSelect.value;
      fillKnowledgeBaseForm(selected);
      setRagPanelStatus('已切换表单到知识库：' + ragKbSelect.value);
    });

    if (ragRefreshBtn) {
      ragRefreshBtn.addEventListener('click', loadKnowledgeBases);
    }

    if (ragSelectBtn) {
      ragSelectBtn.addEventListener('click', function () {
        const knowledgeBase = getCurrentKnowledgeBase();
        if (!knowledgeBase) {
          setRagPanelStatus('请先选择或输入知识库编号', true);
          return;
        }
        requestRag('/api/rag/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ knowledge_base: knowledgeBase })
        }, '当前知识库已切换');
      });
    }

    if (ragSaveBtn) {
      ragSaveBtn.addEventListener('click', function () {
        const payload = buildKbPayload();
        if (!payload.knowledge_base) {
          setRagPanelStatus('请输入知识库编号', true);
          return;
        }
        requestRag('/api/rag/kbs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }, '知识库配置已保存');
      });
    }

    if (ragStatusBtn) {
      ragStatusBtn.addEventListener('click', function () {
        const knowledgeBase = getCurrentKnowledgeBase();
        requestRag('/api/rag/status?knowledge_base=' + encodeURIComponent(knowledgeBase), { method: 'GET' }, '状态已刷新');
      });
    }

    if (ragContentsBtn) {
      ragContentsBtn.addEventListener('click', function () {
        const knowledgeBase = getCurrentKnowledgeBase();
        requestRag('/api/rag/contents?knowledge_base=' + encodeURIComponent(knowledgeBase) + '&limit=12&offset=0', { method: 'GET' }, '库内容已加载');
      });
    }

    if (ragIngestBtn) {
      ragIngestBtn.addEventListener('click', function () {
        const payload = buildKbPayload();
        if (!payload.knowledge_base) {
          setRagPanelStatus('请输入知识库编号', true);
          return;
        }
        requestRag('/api/rag/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            knowledge_base: payload.knowledge_base,
            input_dir: payload.input_dir,
            rebuild: false
          })
        }, '增量入库完成');
      });
    }

    if (ragRebuildBtn) {
      ragRebuildBtn.addEventListener('click', function () {
        const payload = buildKbPayload();
        if (!payload.knowledge_base) {
          setRagPanelStatus('请输入知识库编号', true);
          return;
        }
        const confirmed = window.confirm('确认重建当前知识库索引吗？这会重新执行入库流程。');
        if (!confirmed) {
          setRagPanelStatus('已取消重建');
          return;
        }
        requestRag('/api/rag/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            knowledge_base: payload.knowledge_base,
            input_dir: payload.input_dir,
            rebuild: true
          })
        }, '重建索引完成');
      });
    }

    loadKnowledgeBases();
  }

  sendButton.addEventListener('click', sendCommand);
  userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendCommand();
    }
  });
  loadFileBtn.addEventListener('click', loadFile);
  saveFileBtn.addEventListener('click', saveFile);
  if (clearDisplayBtn) {
    clearDisplayBtn.addEventListener('click', clearDisplayWindow);
  }
  bindRagPanel();
});
