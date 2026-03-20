/**
 * API helpers – fetch + SSE streaming cho RAG Chatbot backend.
 */

const BASE = ""; // proxy qua Vite → localhost:8000

// ── Generic fetch ─────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// ── Upload .docx ──────────────────────────────────────────
export async function uploadDocx(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload thất bại");
  }
  return res.json();
}

// ── Upload folder (.doc/.docx) ────────────────────────────
export async function uploadFolder(files) {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`${BASE}/api/upload-folder`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload folder thất bại");
  }
  return res.json();
}

// ── Datasets ──────────────────────────────────────────────
export async function getDatasets() {
  return apiFetch("/api/datasets");
}

export async function deleteDataset(id) {
  return apiFetch(`/api/datasets/${id}`, { method: "DELETE" });
}

// ── Chat (SSE streaming) ─────────────────────────────────
export async function chatStream(question, temperature, onToken, onSources, onDone) {
  const res = await fetch(`${BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      temperature,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request thất bại");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const payload = trimmed.slice(6);

      if (payload === "[DONE]") {
        onDone?.();
        return;
      }

      try {
        const data = JSON.parse(payload);
        if (data.token) {
          // Check if it's a metadata line (sources, intent, etc.)
          try {
            const inner = JSON.parse(data.token);
            if (inner.type === "sources") {
              onSources?.(inner.data);
              continue;
            }
            if (inner.type === "intent") {
              // Intent metadata – skip, don't display as text
              continue;
            }
          } catch {
            // Not JSON – it's a regular token
          }
          onToken?.(data.token);
        }
      } catch {
        // Ignore parse errors
      }
    }
  }
  onDone?.();
}

// ── GPU status ────────────────────────────────────────────
export async function getGpuStatus() {
  return apiFetch("/api/gpu");
}
