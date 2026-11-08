/* Shared helpers for LightRAG-DuckDB UI */

function showToast(message, kind = "ok") {
  let el = document.querySelector(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.remove("ok", "err");
  el.classList.add(kind, "show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 3500);
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  let data = null;
  try { data = await resp.json(); } catch (e) { /* ignore */ }
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return data;
}

/* Minimal safe markdown renderer (headings, bold, italics, code, lists, links) */
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "", inList = false, inOl = false, inCode = false;
  const closeLists = () => {
    if (inList) { html += "</ul>"; inList = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  for (let line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) { html += "</code></pre>"; inCode = false; }
      else { closeLists(); html += "<pre><code>"; inCode = true; }
      continue;
    }
    if (inCode) { html += line + "\n"; continue; }
    let l = line
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/\*(.+?)\*/g, "<i>$1</i>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[E(\d+)\]/g, '<span class="cite" title="Entity $1">E$1</span>')
      .replace(/\[R(\d+)\]/g, '<span class="cite rel" title="Relationship $1">R$1</span>');
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) { closeLists(); html += `<h${h[1].length + 1}>${h[2]}</h${h[1].length + 1}>`; continue; }
    const ul = l.match(/^\s*[-*]\s+(.*)/);
    if (ul) {
      if (!inList) { closeLists(); html += "<ul>"; inList = true; }
      html += `<li>${ul[1]}</li>`; continue;
    }
    const ol = l.match(/^\s*\d+\.\s+(.*)/);
    if (ol) {
      if (!inOl) { closeLists(); html += "<ol>"; inOl = true; }
      html += `<li>${ol[1]}</li>`; continue;
    }
    closeLists();
    if (l.trim() === "") continue;
    html += `<p>${l}</p>`;
  }
  closeLists();
  if (inCode) html += "</code></pre>";
  return html;
}
