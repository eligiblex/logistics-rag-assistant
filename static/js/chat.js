/* All model/document text is rendered through textContent, never innerHTML. */
const byId = id => document.getElementById(id);
let busy = false;
let refreshTimer = null;

function setBusy(value) {
    busy = value;
    ["send-button", "clear-chat", "new-chat", "refresh-kb"].forEach(id => byId(id).disabled = value);
    byId("typing-indicator").style.display = value ? "inline" : "none";
}
function addMessage(role, text) {
    const box = document.createElement("div");
    box.className = "message " + (role === "user" ? "user-message" : "assistant-message");
    const content = document.createElement("div");
    content.className = "message-content";
    const label = document.createElement("strong");
    label.textContent = role === "user" ? "你" : "物流智能助手";
    const body = document.createElement("div");
    body.className = "message-text";
    body.style.whiteSpace = "pre-wrap";
    body.textContent = text;
    content.append(label, body);
    box.append(content);
    byId("chat-messages").append(box);
    return body;
}
function showSources(sources) {
    const panel = byId("sources-list");
    panel.replaceChildren();
    if (!sources.length) { panel.textContent = "本次检索未返回参考资料"; return; }
    sources.forEach(source => {
        const card = document.createElement("div");
        card.className = "source-item";
        const title = document.createElement("strong");
        title.textContent = "[" + source.citation + "] " + source.source +
            (source.page ? " · 第 " + source.page + " 页" : "") +
            " · 块 " + (source.chunk_id ?? "—");
        const excerpt = document.createElement("p");
        excerpt.style.whiteSpace = "pre-wrap";
        excerpt.textContent = source.content;
        card.append(title, excerpt);
        panel.append(card);
    });
}
async function send() {
    const question = byId("message-input").value.trim();
    if (busy || !question) return;
    if (question.length > 1000) { alert("问题不能超过 1000 字"); return; }
    setBusy(true);
    addMessage("user", question);
    const answer = addMessage("assistant", "");
    byId("message-input").value = "";
    byId("sources-list").textContent = "正在检索...";
    try {
        const streaming = byId("stream-toggle").checked;
        const response = await fetch(streaming ? "/api/query/stream" : "/api/query", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        if (!response.ok) throw new Error("请求失败（HTTP " + response.status + "）");
        if (!streaming) {
            const result = await response.json();
            answer.textContent = result.answer;
            showSources(result.sources);
        } else {
            let complete = false;
            const parser = new SSEParser(event => {
                if (event.type === "sources") showSources(event.sources);
                if (event.type === "token") answer.textContent += event.content;
                if (event.type === "done") complete = true;
                if (event.type === "error") throw new Error(event.message);
                byId("chat-messages").scrollTop = byId("chat-messages").scrollHeight;
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            try {
                while (true) {
                    const {value, done} = await reader.read();
                    if (done) { parser.feed(decoder.decode(), true); break; }
                    parser.feed(decoder.decode(value, {stream: true}));
                }
                if (!complete) throw new Error("回答未完成，连接已中断");
            } finally {
                await reader.cancel().catch(() => {});
                reader.releaseLock();
            }
        }
    } catch (error) {
        answer.textContent += "\n[未完成] " + error.message + "。请重试。";
    } finally { setBusy(false); }
}
async function updateStatus() {
    try {
        const [statusResponse, kbResponse] = await Promise.all([fetch("/api/status"), fetch("/api/knowledge-base/info")]);
        if (!statusResponse.ok || !kbResponse.ok) throw new Error("not ready");
        const status = await statusResponse.json();
        const kb = await kbResponse.json();
        byId("model-info").textContent = status.model;
        byId("kb-info").textContent = kb.document_count + " 个知识片段";
        byId("status").textContent = status.refresh.status === "failed" ? "刷新失败，请检查日志" :
            status.refresh.status === "running" ? "刷新中" : "就绪";
        if (status.refresh.status === "running") {
            clearTimeout(refreshTimer);
            refreshTimer = setTimeout(updateStatus, 1500);
        }
    } catch (_) { byId("status").textContent = "服务未就绪"; }
}
function clearChat() {
    if (busy) return;
    byId("chat-messages").replaceChildren();
    byId("sources-list").textContent = "暂无来源信息";
}
byId("send-button").addEventListener("click", send);
byId("message-input").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
});
document.querySelectorAll(".example-btn").forEach(button => button.addEventListener("click", () => {
    if (!busy) { byId("message-input").value = button.dataset.question; send(); }
}));
byId("clear-chat").addEventListener("click", clearChat);
byId("new-chat").addEventListener("click", clearChat);
byId("refresh-kb").addEventListener("click", async () => {
    if (busy) return;
    byId("refresh-kb").disabled = true;
    try {
        const response = await fetch("/api/knowledge-base/refresh", {method: "POST"});
        if (!response.ok) throw new Error("刷新未启动（HTTP " + response.status + "）");
        byId("status").textContent = "刷新中";
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(updateStatus, 1500);
    } catch (error) { alert(error.message); }
    finally { byId("refresh-kb").disabled = false; }
});
byId("toggle-sources").addEventListener("click", () => {
    const panel = document.querySelector(".sources-content");
    panel.hidden = !panel.hidden;
});
byId("current-time").textContent = new Date().toLocaleTimeString();
updateStatus();
