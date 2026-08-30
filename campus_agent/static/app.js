"use strict";

const SESSION_KEY = "campus-agent-session-id";
const PLANNER_KEY = "campus-agent-planner";

const elements = {
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  messages: document.querySelector("#message-list"),
  template: document.querySelector("#message-template"),
  clear: document.querySelector("#clear-button"),
  plannerOptions: [...document.querySelectorAll(".planner-option")],
  examples: [...document.querySelectorAll(".example-question")],
  suggestions: [...document.querySelectorAll("#suggestion-row button")],
  modelStatus: document.querySelector("#model-status"),
  networkNote: document.querySelector("#network-note"),
  modeLabel: document.querySelector("#chat-mode-label"),
  inputHint: document.querySelector("#input-hint"),
  traceEmpty: document.querySelector("#trace-empty"),
  traceList: document.querySelector("#trace-list"),
  requestTime: document.querySelector("#request-time"),
  metricCalls: document.querySelector("#metric-calls"),
  metricTokens: document.querySelector("#metric-tokens"),
  metricRetries: document.querySelector("#metric-retries"),
  metricFallbacks: document.querySelector("#metric-fallbacks"),
  knowledgeButton: document.querySelector("#knowledge-button"),
  knowledgeDialog: document.querySelector("#knowledge-dialog"),
  knowledgeClose: document.querySelector("#knowledge-close"),
  knowledgeForm: document.querySelector("#knowledge-upload-form"),
  knowledgeFile: document.querySelector("#knowledge-file"),
  knowledgeDropzone: document.querySelector("#knowledge-dropzone"),
  knowledgeSelectedFile: document.querySelector("#knowledge-selected-file"),
  knowledgeUploadButton: document.querySelector("#knowledge-upload-button"),
  knowledgeRebuildButton: document.querySelector("#knowledge-rebuild-button"),
  knowledgeFeedback: document.querySelector("#knowledge-feedback"),
  knowledgeDocumentList: document.querySelector("#knowledge-document-list"),
  knowledgeBuiltInCount: document.querySelector("#knowledge-built-in-count"),
  knowledgeUploadedCount: document.querySelector("#knowledge-uploaded-count"),
  knowledgeChunkCount: document.querySelector("#knowledge-chunk-count"),
  knowledgeIndexVersion: document.querySelector("#knowledge-index-version"),
};

const state = {
  sessionId: getOrCreateSessionId(),
  planner: localStorage.getItem(PLANNER_KEY) === "deepseek" ? "deepseek" : "rule",
  deepseekConfigured: false,
  deepseekConfigError: null,
  deepseekModel: "deepseek-v4-flash",
  businessApiMode: "disabled",
  businessApiNetworkEnabled: false,
  businessApiError: null,
  csrfToken: null,
  knowledgeBusy: false,
  selectedKnowledgeFile: null,
  pending: false,
};

function getOrCreateSessionId() {
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing && /^[A-Za-z0-9_-]{1,64}$/.test(existing)) {
    return existing;
  }
  const randomPart = globalThis.crypto?.randomUUID?.() ??
    `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const sessionId = `web-${randomPart}`;
  localStorage.setItem(SESSION_KEY, sessionId);
  return sessionId;
}

async function apiRequest(url, options = {}) {
  const method = String(options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers ?? {});
  if (
    options.body !== undefined &&
    typeof options.body === "string" &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }
  if (!["GET", "HEAD"].includes(method) && state.csrfToken) {
    headers.set("X-CSRF-Token", state.csrfToken);
  }

  const response = await fetch(url, {
    ...options,
    method,
    headers,
  });
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // 非 JSON 错误响应使用默认消息。
    }
    throw new Error(message);
  }
  return response.json();
}

async function streamApiRequest(url, options, onEvent) {
  const method = String(options.method ?? "POST").toUpperCase();
  const headers = new Headers(options.headers ?? {});
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (state.csrfToken) headers.set("X-CSRF-Token", state.csrfToken);
  const response = await fetch(url, { ...options, method, headers });
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // 使用默认错误信息。
    }
    throw new Error(message);
  }
  if (!response.body) throw new Error("浏览器没有提供可读取的响应流");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let transportDone = false;
  const maxBufferCharacters = 1024 * 1024;

  const dispatchBlock = async (rawBlock) => {
    const block = rawBlock.replaceAll("\r\n", "\n").trim();
    if (!block || block.startsWith(":")) return;
    let eventName = "message";
    const dataLines = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    let payload;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      throw new Error("服务端返回了无效的流式数据");
    }
    await onEvent(eventName, payload);
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      if (buffer.length > maxBufferCharacters) {
        throw new Error("服务端流式事件超过安全大小限制");
      }
      buffer = buffer.replaceAll("\r\n", "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        await dispatchBlock(block);
        boundary = buffer.indexOf("\n\n");
      }
      if (done) {
        transportDone = true;
        break;
      }
    }
    if (buffer.trim()) await dispatchBlock(buffer);
  } finally {
    if (!transportDone) {
      try {
        await reader.cancel();
      } catch {
        // 连接已断开时 cancel 可能再次失败。
      }
    }
    reader.releaseLock();
  }
}

function renderMessage(role, content, options = {}) {
  const fragment = elements.template.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const avatar = fragment.querySelector(".message-avatar");
  const meta = fragment.querySelector(".message-meta");
  const bubble = fragment.querySelector(".message-bubble");

  article.classList.add(role === "user" ? "user" : "assistant");
  if (options.error) article.classList.add("error");
  avatar.textContent = role === "user" ? "你" : "CA";
  meta.textContent = role === "user" ? "你" : "Campus Agent";
  bubble.textContent = content;
  if (role !== "user" && Array.isArray(options.citations) && options.citations.length) {
    const citationList = document.createElement("div");
    citationList.className = "message-citations";
    for (const citation of options.citations) {
      const item = document.createElement("span");
      item.className = "message-citation";
      const citationId = String(citation.citation_id ?? "来源");
      const source = String(citation.source ?? "知识库");
      const title = String(citation.title ?? "资料片段");
      item.textContent = `[${citationId}] ${source} · ${title}`;
      item.title = String(citation.snippet ?? "");
      citationList.appendChild(item);
    }
    bubble.insertAdjacentElement("afterend", citationList);
  }
  elements.messages.appendChild(fragment);
  scrollMessagesToBottom();
  return elements.messages.lastElementChild;
}

function renderWelcome() {
  renderMessage(
    "assistant",
    "你好，我是 Campus Agent。可以查询课程安排，也可以从内置资料或你上传的知识库中检索校园办事流程。\n\n你可以先试试：告诉我周三的课程和校园卡补办流程。",
  );
}

function renderLoading() {
  const fragment = elements.template.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const avatar = fragment.querySelector(".message-avatar");
  const meta = fragment.querySelector(".message-meta");
  const bubble = fragment.querySelector(".message-bubble");
  article.classList.add("assistant", "loading");
  avatar.textContent = "CA";
  meta.textContent = "正在规划并调用工具";
  bubble.replaceChildren();
  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("i");
    dot.className = "typing-dot";
    bubble.appendChild(dot);
  }
  elements.messages.appendChild(fragment);
  scrollMessagesToBottom();
  return elements.messages.lastElementChild;
}

function scrollMessagesToBottom() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderTrace(events) {
  elements.traceList.replaceChildren();
  if (!Array.isArray(events) || events.length === 0) {
    elements.traceEmpty.hidden = false;
    elements.traceList.hidden = true;
    return;
  }

  const labels = {
    plan: ["P", "规划任务"],
    retrieve: ["R", "检索知识库"],
    tool_result: ["T", "执行工具"],
    grade: ["G", "评估资料"],
    rewrite: ["Q", "改写问题"],
    verify: ["V", "验证回答"],
    final: ["A", "生成回答"],
  };
  for (const event of events) {
    const [markerText, label] = labels[event.type] ?? ["·", "运行事件"];
    const item = document.createElement("li");
    item.className = `trace-item ${event.type ?? ""}`;
    const marker = document.createElement("span");
    marker.className = "trace-marker";
    marker.textContent = markerText;
    const title = document.createElement("strong");
    title.textContent = label;
    const detail = document.createElement("p");
    detail.textContent = String(event.detail ?? "");
    item.append(marker, title, detail);
    elements.traceList.appendChild(item);
  }
  elements.traceEmpty.hidden = true;
  elements.traceList.hidden = false;
}

function renderMetrics(metrics = {}, elapsedMs = null) {
  const api = metrics.api && typeof metrics.api === "object" ? metrics.api : {};
  elements.metricCalls.textContent = String(api.calls ?? metrics.network_calls ?? 0);
  elements.metricTokens.textContent = Number(api.total_tokens ?? 0).toLocaleString("zh-CN");
  elements.metricRetries.textContent = String(api.retries ?? 0);
  elements.metricFallbacks.textContent = String(metrics.fallback_count ?? 0);
  if (elapsedMs !== null) {
    elements.requestTime.textContent = `${Math.round(elapsedMs)} ms`;
  }
}

function updatePlannerUI() {
  for (const option of elements.plannerOptions) {
    option.classList.toggle("active", option.dataset.planner === state.planner);
    option.setAttribute("aria-pressed", String(option.dataset.planner === state.planner));
  }

  if (state.planner === "rule") {
    elements.modeLabel.textContent = state.businessApiNetworkEnabled
      ? "规则模式 · 可调用授权业务 API"
      : "规则模式 · 本地运行";
    elements.inputHint.textContent = "本地课程 + RAG + 只读业务状态";
    elements.networkNote.classList.remove("deepseek-ready");
    elements.networkNote.querySelector("strong").textContent =
      state.businessApiNetworkEnabled ? "可能调用校园业务 API" : "不调用大模型";
    elements.networkNote.querySelector("span:last-child").textContent =
      state.businessApiMode === "mock"
        ? "实时状态使用本地模拟数据，并会在答案中明确标注。"
        : state.businessApiNetworkEnabled
          ? "仅实时状态由 Python 后端访问已授权的只读校园接口。"
          : state.businessApiError || "课程和知识检索均在本机完成。";
  } else if (state.deepseekConfigured) {
    elements.modeLabel.textContent = `DeepSeek · ${state.deepseekModel}`;
    elements.inputHint.textContent = "DeepSeek 规划 + 本地工具与 RAG";
    elements.networkNote.classList.add("deepseek-ready");
    elements.networkNote.querySelector("strong").textContent = "将调用 DeepSeek API";
    elements.networkNote.querySelector("span:last-child").textContent =
      state.businessApiNetworkEnabled
        ? "后端发送问题与工具上下文；实时状态还可能访问已授权校园接口。"
        : "后端发送问题与工具上下文，课程、知识检索和 Mock 业务数据仍在本机。";
  } else {
    elements.modeLabel.textContent = state.deepseekConfigError
      ? "DeepSeek 配置有误 · 自动降级"
      : "DeepSeek 未配置 · 自动降级";
    elements.inputHint.textContent = state.deepseekConfigError
      ? "在线配置无效，将使用规则规划器"
      : "未检测到 API Key，将使用规则规划器";
    elements.networkNote.classList.remove("deepseek-ready");
    elements.networkNote.querySelector("strong").textContent =
      state.businessApiNetworkEnabled
        ? "DeepSeek 已降级，仅业务 API 可能联网"
        : state.deepseekConfigError
          ? "DeepSeek 配置有误"
          : "API Key 未配置";
    elements.networkNote.querySelector("span:last-child").textContent =
      state.businessApiNetworkEnabled
        ? "模型请求会回退规则规划器；只有实时状态查询可能访问授权接口。"
        : state.deepseekConfigError ||
          "不会访问网络，本次请求会安全回退到规则规划器。";
  }
}

async function selectPlanner(planner) {
  if (state.pending) return;
  state.planner = planner;
  localStorage.setItem(PLANNER_KEY, planner);
  updatePlannerUI();
  try {
    const metrics = await apiRequest(`/api/metrics?planner=${encodeURIComponent(planner)}`);
    renderMetrics(metrics);
  } catch {
    // 模式切换不应因指标接口失败而中断。
  }
}

async function sendMessage(query) {
  const cleanedQuery = query.trim();
  if (!cleanedQuery || state.pending) return;

  state.pending = true;
  elements.send.disabled = true;
  for (const option of elements.plannerOptions) option.disabled = true;
  elements.input.value = "";
  resizeTextarea();
  renderMessage("user", cleanedQuery);
  const loadingMessage = renderLoading();
  const startedAt = performance.now();

  try {
    const traceEvents = [];
    let resultPayload = null;
    await streamApiRequest("/api/chat/stream", {
      method: "POST",
      body: JSON.stringify({
        query: cleanedQuery,
        session_id: state.sessionId,
        planner: state.planner,
      }),
    }, async (eventName, payload) => {
      if (eventName === "trace") {
        traceEvents.push(payload);
        renderTrace(traceEvents);
      } else if (eventName === "result") {
        resultPayload = payload;
        if (loadingMessage.isConnected) loadingMessage.remove();
        renderMessage("assistant", payload.answer, { citations: payload.citations });
        renderTrace(payload.events);
        if (payload.planner === state.planner) {
          renderMetrics(
            payload.metrics,
            payload.elapsed_ms ?? performance.now() - startedAt,
          );
        }
      } else if (eventName === "error") {
        throw new Error(String(payload.detail ?? "Agent 流式执行失败"));
      }
    });
    if (!resultPayload) throw new Error("连接结束前没有收到最终回答");
  } catch (error) {
    if (loadingMessage.isConnected) loadingMessage.remove();
    // POST 不能自动重试；重新读取历史以避免“服务端已提交、浏览器没收到”造成错位。
    await loadHistory();
    renderMessage(
      "assistant",
      `连接未完整结束，已同步服务端历史：${error instanceof Error ? error.message : "未知错误"}`,
      { error: true },
    );
    elements.requestTime.textContent = "失败";
  } finally {
    state.pending = false;
    elements.send.disabled = false;
    for (const option of elements.plannerOptions) option.disabled = false;
    elements.input.focus();
  }
}

function formatBytes(value) {
  const bytes = Number(value ?? 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

function renderKnowledgeStats(stats) {
  if (!stats || typeof stats !== "object") return;
  elements.knowledgeBuiltInCount.textContent = String(stats.built_in_documents ?? 0);
  elements.knowledgeUploadedCount.textContent = String(stats.uploaded_documents ?? 0);
  elements.knowledgeChunkCount.textContent = String(stats.chunk_count ?? 0);
  elements.knowledgeIndexVersion.textContent = `v${stats.index_version ?? 0}`;
}

function setKnowledgeFeedback(message = "", type = "") {
  elements.knowledgeFeedback.textContent = message;
  elements.knowledgeFeedback.className = "knowledge-feedback";
  if (type) elements.knowledgeFeedback.classList.add(type);
}

function setKnowledgeBusy(busy) {
  state.knowledgeBusy = busy;
  elements.knowledgeUploadButton.disabled = busy;
  elements.knowledgeRebuildButton.disabled = busy;
  elements.knowledgeFile.disabled = busy;
  for (const button of elements.knowledgeDocumentList.querySelectorAll("button")) {
    button.disabled = busy;
  }
}

function setSelectedKnowledgeFile(file) {
  state.selectedKnowledgeFile = file ?? null;
  elements.knowledgeSelectedFile.textContent = file
    ? `${file.name} · ${formatBytes(file.size)}`
    : "尚未选择文件";
}

function renderKnowledgeDocuments(documents) {
  elements.knowledgeDocumentList.replaceChildren();
  if (!Array.isArray(documents) || documents.length === 0) {
    const empty = document.createElement("div");
    empty.className = "knowledge-list-empty";
    empty.textContent = "还没有上传资料。导入一份课程说明或办事指南后，就可以直接向 Agent 提问。";
    elements.knowledgeDocumentList.appendChild(empty);
    return;
  }

  for (const documentItem of documents) {
    const item = document.createElement("article");
    item.className = "knowledge-document-item";

    const info = document.createElement("div");
    info.className = "knowledge-document-info";
    const title = document.createElement("strong");
    title.textContent = String(documentItem.display_name ?? "未命名资料");
    const metadata = document.createElement("span");
    metadata.className = "knowledge-document-meta";
    const createdAt = documentItem.created_at
      ? new Date(documentItem.created_at).toLocaleString("zh-CN", { hour12: false })
      : "时间未知";
    metadata.textContent = `${formatBytes(documentItem.size_bytes)} · ${documentItem.chunk_count ?? 0} 个片段 · ${createdAt}`;
    info.append(title, metadata);

    const remove = document.createElement("button");
    remove.className = "knowledge-delete-button";
    remove.type = "button";
    remove.textContent = "删除";
    remove.addEventListener("click", () => {
      deleteKnowledgeDocument(
        String(documentItem.document_id ?? ""),
        String(documentItem.display_name ?? "这份资料"),
      );
    });
    item.append(info, remove);
    elements.knowledgeDocumentList.appendChild(item);
  }
}

async function loadKnowledgeDocuments() {
  elements.knowledgeDocumentList.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "knowledge-list-empty";
  loading.textContent = "正在读取知识库…";
  elements.knowledgeDocumentList.appendChild(loading);
  try {
    const payload = await apiRequest("/api/knowledge/documents");
    renderKnowledgeStats(payload.stats);
    renderKnowledgeDocuments(payload.documents);
  } catch (error) {
    renderKnowledgeDocuments([]);
    setKnowledgeFeedback(
      `读取失败：${error instanceof Error ? error.message : "未知错误"}`,
      "error",
    );
  }
}

async function openKnowledgeDialog() {
  if (!elements.knowledgeDialog.open) elements.knowledgeDialog.showModal();
  document.body.style.overflow = "hidden";
  setKnowledgeFeedback();
  await loadKnowledgeDocuments();
}

async function uploadKnowledgeDocument(event) {
  event.preventDefault();
  if (state.knowledgeBusy) return;
  const file = state.selectedKnowledgeFile;
  if (!file) {
    setKnowledgeFeedback("请先选择一个 .md、.txt 或 .pdf 文件。", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    setKnowledgeFeedback("单个文件不能超过 10 MiB。", "error");
    return;
  }
  if (!/\.(md|txt|pdf)$/i.test(file.name)) {
    setKnowledgeFeedback("仅支持 .md、.txt 和 .pdf 文件。", "error");
    return;
  }

  setKnowledgeBusy(true);
  setKnowledgeFeedback(`正在解析并索引 ${file.name}…`);
  try {
    const payload = await apiRequest(
      `/api/knowledge/documents?filename=${encodeURIComponent(file.name)}`,
      {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      },
    );
    renderKnowledgeStats(payload.stats);
    setKnowledgeFeedback(`${file.name} 已进入检索索引，无需重启服务。`, "success");
    elements.knowledgeFile.value = "";
    setSelectedKnowledgeFile(null);
    await loadKnowledgeDocuments();
  } catch (error) {
    setKnowledgeFeedback(
      `上传失败：${error instanceof Error ? error.message : "未知错误"}`,
      "error",
    );
  } finally {
    setKnowledgeBusy(false);
  }
}

async function deleteKnowledgeDocument(documentId, displayName) {
  if (state.knowledgeBusy || !documentId) return;
  if (!globalThis.confirm(`确定永久删除“${displayName}”并更新检索索引吗？`)) return;

  setKnowledgeBusy(true);
  setKnowledgeFeedback(`正在删除 ${displayName}…`);
  try {
    const payload = await apiRequest(
      `/api/knowledge/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" },
    );
    renderKnowledgeStats(payload.stats);
    setKnowledgeFeedback(`${displayName} 已删除，旧内容不再参与检索。`, "success");
    await loadKnowledgeDocuments();
  } catch (error) {
    setKnowledgeFeedback(
      `删除失败：${error instanceof Error ? error.message : "未知错误"}`,
      "error",
    );
  } finally {
    setKnowledgeBusy(false);
  }
}

async function rebuildKnowledgeIndex() {
  if (state.knowledgeBusy) return;
  setKnowledgeBusy(true);
  setKnowledgeFeedback("正在从磁盘重新构建检索索引…");
  try {
    const stats = await apiRequest("/api/knowledge/rebuild", { method: "POST" });
    renderKnowledgeStats(stats);
    setKnowledgeFeedback(`索引已重建，当前版本 v${stats.index_version}。`, "success");
    await loadKnowledgeDocuments();
  } catch (error) {
    setKnowledgeFeedback(
      `重建失败：${error instanceof Error ? error.message : "未知错误"}`,
      "error",
    );
  } finally {
    setKnowledgeBusy(false);
  }
}

async function loadHealth() {
  try {
    const health = await apiRequest("/api/health");
    state.csrfToken = typeof health.csrf_token === "string" ? health.csrf_token : null;
    state.deepseekConfigured = Boolean(health.deepseek_configured);
    state.deepseekConfigError = health.deepseek_configuration_error || null;
    state.deepseekModel = health.deepseek_model || state.deepseekModel;
    state.businessApiMode = health.campus_business_api?.mode || "disabled";
    state.businessApiNetworkEnabled = Boolean(
      health.campus_business_api?.network_enabled,
    );
    state.businessApiError = health.campus_business_api?.error || null;
    renderKnowledgeStats(health.knowledge);
    elements.modelStatus.textContent = state.deepseekConfigured
      ? `${state.deepseekModel} 已配置`
      : state.deepseekConfigError
        ? "DeepSeek 配置有误 · 已离线降级"
        : "DeepSeek 未配置 · 可离线运行";
  } catch {
    elements.modelStatus.textContent = "状态检查失败";
  }
  updatePlannerUI();
}

async function loadHistory() {
  try {
    const payload = await apiRequest(
      `/api/history?session_id=${encodeURIComponent(state.sessionId)}`,
    );
    elements.messages.replaceChildren();
    if (!payload.messages?.length) {
      renderWelcome();
      return;
    }
    for (const message of payload.messages) {
      renderMessage(message.role, message.content);
    }
  } catch (error) {
    elements.messages.replaceChildren();
    renderMessage(
      "assistant",
      `无法读取会话：${error instanceof Error ? error.message : "未知错误"}`,
      { error: true },
    );
  }
}

async function clearHistory() {
  if (state.pending) return;
  try {
    await apiRequest(`/api/history?session_id=${encodeURIComponent(state.sessionId)}`, {
      method: "DELETE",
    });
    elements.messages.replaceChildren();
    renderWelcome();
    renderTrace([]);
    elements.requestTime.textContent = "— ms";
  } catch (error) {
    renderMessage(
      "assistant",
      `无法清空会话：${error instanceof Error ? error.message : "未知错误"}`,
      { error: true },
    );
  }
}

function resizeTextarea() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 130)}px`;
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(elements.input.value);
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(elements.input.value);
  }
});

elements.input.addEventListener("input", resizeTextarea);
elements.clear.addEventListener("click", clearHistory);
elements.knowledgeButton.addEventListener("click", openKnowledgeDialog);
elements.knowledgeClose.addEventListener("click", () => elements.knowledgeDialog.close());
elements.knowledgeDialog.addEventListener("close", () => {
  document.body.style.overflow = "";
});
elements.knowledgeDialog.addEventListener("click", (event) => {
  if (event.target === elements.knowledgeDialog) elements.knowledgeDialog.close();
});
elements.knowledgeForm.addEventListener("submit", uploadKnowledgeDocument);
elements.knowledgeRebuildButton.addEventListener("click", rebuildKnowledgeIndex);
elements.knowledgeFile.addEventListener("change", () => {
  setSelectedKnowledgeFile(elements.knowledgeFile.files?.[0] ?? null);
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.knowledgeDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!state.knowledgeBusy) elements.knowledgeDropzone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.knowledgeDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.knowledgeDropzone.classList.remove("dragging");
  });
}

elements.knowledgeDropzone.addEventListener("drop", (event) => {
  if (state.knowledgeBusy) return;
  setSelectedKnowledgeFile(event.dataTransfer?.files?.[0] ?? null);
});

for (const option of elements.plannerOptions) {
  option.addEventListener("click", () => selectPlanner(option.dataset.planner));
}

for (const button of [...elements.examples, ...elements.suggestions]) {
  button.addEventListener("click", () => {
    const question = button.textContent.replace(/^\s*\d{2}\s*/, "").trim();
    elements.input.value = question;
    resizeTextarea();
    elements.input.focus();
  });
}

async function initialize() {
  await loadHealth();
  await loadHistory();
  await selectPlanner(state.planner);
  elements.input.focus();
}

initialize();
