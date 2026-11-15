// free-model-aggregator-02: one server serving the OpenAI Chat Completions and
// Anthropic Messages APIs, answering every prompt with a REAL completion from a
// live free OpenRouter model dialled through the grader's gateway.
//
// Node 22+, standard library only (node:http + global fetch). No dependencies.
// Deterministic: every choice derives from the brief, the request and the
// gateway's answers. No wall clock, no unseeded randomness.

import http from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BriefModel = {
  id: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
};

type Catalog = {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
};

type FailureRow = { status: number; code: string; model: string | null };

type GatewayOk = { ok: true; status: number; json: any; model: string };
type GatewayFail = {
  ok: false;
  status: number;
  code: string;
  message: string;
  model: string;
};
type GatewayResult = GatewayOk | GatewayFail;

// ---------------------------------------------------------------------------
// Brief parsing
// ---------------------------------------------------------------------------

// Extract every top-level JSON object embedded in the brief text, tolerating
// surrounding prose. Braces inside strings are skipped.
function extractJsonObjects(text: string): any[] {
  const out: any[] = [];
  let depth = 0;
  let start = -1;
  let inStr = false;
  let esc = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
      continue;
    }
    if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        try {
          out.push(JSON.parse(text.slice(start, i + 1)));
        } catch {
          // not a JSON object; ignore
        }
        start = -1;
      }
      if (depth < 0) {
        depth = 0;
        start = -1;
      }
    }
  }
  return out;
}

function loadBrief(): { models: BriefModel[] } {
  const file = process.env.MAKEFUN_PROMPT_FILE || "";
  if (!file) return { models: [] };
  let text = "";
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return { models: [] };
  }
  const models: BriefModel[] = [];
  for (const obj of extractJsonObjects(text)) {
    if (obj && Array.isArray(obj.models)) {
      for (const m of obj.models) {
        if (m && typeof m.id === "string" && m.free !== false) {
          models.push({
            id: m.id,
            rating_milli: typeof m.rating_milli === "number" ? m.rating_milli : 0,
            context_length:
              typeof m.context_length === "number" ? m.context_length : null,
            tools: m.tools === true,
            vision: m.vision === true,
            json: m.json === true,
            free: true,
          });
        }
      }
    }
  }
  // De-duplicate by id, keeping first occurrence (brief order).
  const seen = new Set<string>();
  return { models: models.filter((m) => (seen.has(m.id) ? false : (seen.add(m.id), true))) };
}

function catalogFrom(models: BriefModel[]): Catalog {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({
          id: m.id,
          free: true,
          context_length: m.context_length,
        })),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

type Requirements = { json: boolean; tools: boolean; vision: boolean };

function estimateTokens(messages: any[]): number {
  let chars = 0;
  const walk = (v: any): void => {
    if (typeof v === "string") chars += v.length;
    else if (Array.isArray(v)) v.forEach(walk);
    else if (v && typeof v === "object") Object.values(v).forEach(walk);
  };
  walk(messages);
  return Math.ceil(chars / 4) + 64;
}

// Eligible models, best rating first, id ascending as the deterministic tie-break.
function rankCandidates(models: BriefModel[], req: Requirements, estTokens: number): BriefModel[] {
  const fits = (m: BriefModel): boolean => {
    if (req.json && !m.json) return false;
    if (req.tools && !m.tools) return false;
    if (req.vision && !m.vision) return false;
    return true;
  };
  let pool = models.filter(fits);
  if (pool.length === 0) pool = models.filter((m) => (!req.json || m.json) && (!req.tools || m.tools));
  if (pool.length === 0) pool = models.slice();
  const withRoom = pool.filter(
    (m) => m.context_length === null || m.context_length >= estTokens,
  );
  if (withRoom.length > 0) pool = withRoom;
  return pool
    .slice()
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function textOf(content: any): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => p && p.type === "text" && typeof p.text === "string")
      .map((p) => p.text)
      .join("");
  }
  return "";
}

function hasVision(messages: any[]): boolean {
  let found = false;
  const walk = (v: any): void => {
    if (found) return;
    if (Array.isArray(v)) {
      for (const item of v) walk(item);
      return;
    }
    if (v && typeof v === "object") {
      if (v.type === "image_url" || v.type === "image") found = true;
      for (const val of Object.values(v)) walk(val);
    }
  };
  walk(messages);
  return found;
}

function estimateOutputTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

// ---------------------------------------------------------------------------
// Anthropic <-> OpenAI translation
// ---------------------------------------------------------------------------

function anthropicSystemToMessages(body: any): any[] {
  const out: any[] = [];
  const sys = body.system;
  if (typeof sys === "string" && sys) {
    out.push({ role: "system", content: sys });
  } else if (Array.isArray(sys)) {
    const t = sys
      .filter((b: any) => b && b.type === "text" && typeof b.text === "string")
      .map((b: any) => b.text)
      .join("");
    if (t) out.push({ role: "system", content: t });
  }
  return out;
}

function anthropicMessagesToOpenAI(body: any): any[] {
  const messages: any[] = anthropicSystemToMessages(body);
  const input = Array.isArray(body.messages) ? body.messages : [];
  for (const m of input) {
    const role = m.role;
    const blocks = Array.isArray(m.content)
      ? m.content
      : [{ type: "text", text: typeof m.content === "string" ? m.content : "" }];
    if (role === "user") {
      const toolResults = blocks.filter((b: any) => b && b.type === "tool_result");
      const parts: any[] = [];
      for (const b of blocks) {
        if (!b || typeof b !== "object") continue;
        if (b.type === "text" && typeof b.text === "string") {
          parts.push({ type: "text", text: b.text });
        } else if (b.type === "image" && b.source) {
          const src = b.source;
          let url = "";
          if (src.type === "base64" && src.data) {
            url = `data:${src.media_type || "image/png"};base64,${src.data}`;
          } else if (typeof src.url === "string") {
            url = src.url;
          }
          parts.push({ type: "image_url", image_url: { url } });
        }
      }
      if (parts.length === 1 && parts[0].type === "text") {
        messages.push({ role: "user", content: parts[0].text });
      } else if (parts.length > 0) {
        messages.push({ role: "user", content: parts });
      } else if (toolResults.length === 0) {
        messages.push({ role: "user", content: "" });
      }
      for (const tr of toolResults) {
        const c =
          typeof tr.content === "string"
            ? tr.content
            : JSON.stringify(tr.content === undefined ? "" : tr.content);
        messages.push({ role: "tool", tool_call_id: tr.tool_use_id || "", content: c });
      }
    } else if (role === "assistant") {
      const text = blocks
        .filter((b: any) => b && b.type === "text" && typeof b.text === "string")
        .map((b: any) => b.text)
        .join("");
      const toolCalls = blocks
        .filter((b: any) => b && b.type === "tool_use")
        .map((b: any) => ({
          id: b.id || "",
          type: "function",
          function: {
            name: b.name || "",
            arguments: JSON.stringify(b.input === undefined ? {} : b.input),
          },
        }));
      const msg: any = { role: "assistant" };
      if (text) msg.content = text;
      else if (toolCalls.length > 0) msg.content = null;
      else msg.content = "";
      if (toolCalls.length > 0) msg.tool_calls = toolCalls;
      messages.push(msg);
    }
  }
  return messages;
}

function anthropicToolsToOpenAI(tools: any): any[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  return tools.map((t: any) => ({
    type: "function",
    function: {
      name: t.name || "",
      description: t.description || "",
      parameters: t.input_schema || { type: "object", properties: {} },
    },
  }));
}

function anthropicToolChoiceToOpenAI(tc: any): any {
  if (!tc || typeof tc !== "object") return undefined;
  if (tc.type === "auto" || tc.type === "none") return tc.type;
  if (tc.type === "any") return "required";
  if (tc.type === "tool" && typeof tc.name === "string") {
    return { type: "function", function: { name: tc.name } };
  }
  return undefined;
}

function mapAnthropicStop(finish: string): string {
  if (finish === "tool_calls") return "tool_use";
  if (finish === "length") return "max_tokens";
  return "end_turn";
}

let anthropicCounter = 0;
let openaiCounter = 0;

function openaiToAnthropic(gw: any, dialedId: string): any {
  const choice = (gw.choices && gw.choices[0]) || {};
  const msg = choice.message || {};
  const text = textOf(msg.content);
  const content: any[] = [];
  if (text) content.push({ type: "text", text });
  for (const tc of msg.tool_calls || []) {
    let input: any = {};
    try {
      input = JSON.parse((tc.function && tc.function.arguments) || "{}");
    } catch {
      input = {};
    }
    content.push({
      type: "tool_use",
      id: tc.id || "",
      name: (tc.function && tc.function.name) || "",
      input,
    });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const usage = gw.usage || {};
  // The Messages wire needs an output count of at least 1.
  const outTok = Math.max(
    1,
    typeof usage.completion_tokens === "number"
      ? usage.completion_tokens
      : estimateOutputTokens(text),
  );
  anthropicCounter++;
  return {
    id: "msg_" + String(anthropicCounter).padStart(6, "0"),
    type: "message",
    role: "assistant",
    content,
    model: dialedId,
    stop_reason: mapAnthropicStop(choice.finish_reason || "stop"),
    usage: {
      input_tokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
      output_tokens: outTok,
    },
  };
}

// ---------------------------------------------------------------------------
// Gateway dialling with retry + failover
// ---------------------------------------------------------------------------

function gatewayBase(): string {
  const raw = (process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1").replace(/\/+$/, "");
  return raw;
}

function gatewayKey(): string {
  return process.env.OPENAI_API_KEY || "";
}

function errorCodeFromBody(text: string, status: number): { code: string; message: string } {
  try {
    const j = JSON.parse(text);
    const err = j.error || j;
    if (err && typeof err === "object") {
      return {
        code: typeof err.code === "string" ? err.code : `http_${status}`,
        message: typeof err.message === "string" ? err.message : text.slice(0, 500),
      };
    }
  } catch {
    // not JSON
  }
  return { code: `http_${status}`, message: text.slice(0, 500) || `HTTP ${status}` };
}

async function dialOnce(
  gatewayBody: any,
  model: string,
  stream: boolean,
): Promise<
  | { kind: "ok-json"; json: any }
  | { kind: "ok-stream"; res: any }
  | { kind: "fail"; fail: GatewayFail }
> {
  const body = { ...gatewayBody, model, stream };
  let res: any;
  try {
    res = await fetch(gatewayBase() + "/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + gatewayKey(),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(25000),
    });
  } catch (e: any) {
    const code = e && e.name === "TimeoutError" ? "gateway_timeout" : "gateway_error";
    return {
      kind: "fail",
      fail: { ok: false, status: 502, code, message: String((e && e.message) || e), model },
    };
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const parsed = errorCodeFromBody(text, res.status);
    return {
      kind: "fail",
      fail: { ok: false, status: res.status, code: parsed.code, message: parsed.message, model },
    };
  }
  if (stream) return { kind: "ok-stream", res };
  let json: any;
  try {
    json = await res.json();
  } catch {
    return {
      kind: "fail",
      fail: { ok: false, status: 502, code: "invalid_response", message: "gateway returned invalid JSON", model },
    };
  }
  return { kind: "ok-json", json };
}

// Attempt order: best model first; a 429/5xx earns one same-model retry, any
// other 4xx moves straight to the next-best candidate (retrying it would fail
// the same way). At most 4 gateway calls per prompt. A 401 means no key at
// all: fail over is pointless, stop after the first refusal.
async function dialWithFailover(
  gatewayBody: any,
  candidates: BriefModel[],
  stream: boolean,
): Promise<
  | { ok: true; model: string; json?: any; res?: any; failures: FailureRow[] }
  | { ok: false; failures: FailureRow[]; last: GatewayFail }
> {
  const failures: FailureRow[] = [];
  let last: GatewayFail | null = null;
  let attempts = 0;
  if (candidates.length === 0) {
    // No dialable model in the brief: report it instead of returning a
    // null `last` that would crash the error path.
    return {
      ok: false,
      failures,
      last: { ok: false, status: 503, code: "no_models", message: "no free models in brief", model: "" },
    };
  }
  for (let ci = 0; ci < candidates.length && attempts < 4; ci++) {
    const model = candidates[ci].id;
    let retried = false;
    for (;;) {
      attempts++;
      const attempt = await dialOnce(gatewayBody, model, stream);
      if (attempt.kind === "ok-json") {
        return { ok: true, model, json: attempt.json, failures };
      }
      if (attempt.kind === "ok-stream") {
        return { ok: true, model, res: attempt.res, failures };
      }
      const f = attempt.fail;
      last = f;
      failures.push({ status: f.status, code: f.code, model: f.model });
      if (f.status === 401) return { ok: false, failures, last: last as GatewayFail };
      // Retry the same model once, only on a refusal a retry can cure.
      if ((f.status === 429 || f.status >= 500) && !retried && attempts < 4) {
        retried = true;
        continue;
      }
      break;
    }
  }
  return { ok: false, failures, last: last as GatewayFail };
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

// Rewrite the model field of every OpenAI SSE data chunk, forwarding [DONE]
// untouched. Reads the gateway stream, writes OpenAI SSE to `writeChunk`.
async function forwardOpenAIStream(
  res: any,
  dialedId: string,
  writeChunk: (text: string) => void,
): Promise<{ finishReason: string; usage: any; text: string }> {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finishReason = "";
  let usage: any = null;
  let text = "";
  const handleEvent = (eventText: string): void => {
    const lines = eventText.split("\n");
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      // Trim both ends: a CRLF gateway leaves a trailing \r that would
      // break the "[DONE]" comparison and the JSON parse otherwise.
      const data = line.slice(5).trim();
      if (data === "[DONE]") {
        writeChunk("data: [DONE]\n\n");
        continue;
      }
      if (!data) continue;
      let chunk: any;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }
      chunk.model = dialedId;
      const choice = (chunk.choices && chunk.choices[0]) || {};
      if (choice.finish_reason) finishReason = choice.finish_reason;
      if (chunk.usage) usage = chunk.usage;
      if (choice.delta && typeof choice.delta.content === "string") {
        text += choice.delta.content;
      }
      writeChunk("data: " + JSON.stringify(chunk) + "\n\n");
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // Normalise CRLF to LF so event boundaries ("\n\n") are found even
    // when the gateway frames SSE with \r\n line endings.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx = buffer.indexOf("\n\n");
    while (idx >= 0) {
      const eventText = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      handleEvent(eventText);
      idx = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleEvent(buffer);
  return { finishReason, usage, text };
}

// Translate an OpenAI SSE stream into Anthropic SSE events, in the order the
// Messages API expects: message_start, content_block_start, content_block_delta
// (s), content_block_stop, message_delta (with usage), message_stop.
async function forwardAnthropicStream(
  res: any,
  dialedId: string,
  writeEvent: (event: string, data: any) => void,
): Promise<void> {
  anthropicCounter++;
  const msgId = "msg_" + String(anthropicCounter).padStart(6, "0");
  writeEvent("message_start", {
    type: "message_start",
    message: {
      id: msgId,
      type: "message",
      role: "assistant",
      content: [],
      model: dialedId,
      stop_reason: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finishReason = "";
  let usage: any = null;
  let text = "";
  let nextBlock = 0;
  let textBlock = -1;
  let lastBlock = -1;
  const toolBlocks = new Map<number, number>(); // tool_call index -> block index
  const openBlocks: number[] = [];
  const startBlock = (index: number, block: any): void => {
    writeEvent("content_block_start", { type: "content_block_start", index, content_block: block });
    openBlocks.push(index);
    lastBlock = index;
  };
  const handleChunk = (chunk: any): void => {
    const choice = (chunk.choices && chunk.choices[0]) || {};
    if (choice.finish_reason) finishReason = choice.finish_reason;
    if (chunk.usage) usage = chunk.usage;
    const delta = choice.delta || {};
    if (typeof delta.content === "string" && delta.content) {
      // Text arriving after a tool block has started belongs in a new
      // text block: Anthropic deltas run in content-block order. Once
      // the new block is current, further text deltas keep appending
      // to it (lastBlock === textBlock).
      if (textBlock < 0 || lastBlock !== textBlock) {
        textBlock = nextBlock++;
        startBlock(textBlock, { type: "text", text: "" });
      }
      text += delta.content;
      writeEvent("content_block_delta", {
        type: "content_block_delta",
        index: textBlock,
        delta: { type: "text_delta", text: delta.content },
      });
    }
    const toolCalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const tc of toolCalls) {
      const tIndex = typeof tc.index === "number" ? tc.index : 0;
      let blockIndex = toolBlocks.get(tIndex);
      if (blockIndex === undefined) {
        blockIndex = nextBlock++;
        toolBlocks.set(tIndex, blockIndex);
        startBlock(blockIndex, {
          type: "tool_use",
          id: tc.id || "",
          name: (tc.function && tc.function.name) || "",
          input: {},
        });
      }
      const args = tc.function && tc.function.arguments;
      if (typeof args === "string" && args) {
        writeEvent("content_block_delta", {
          type: "content_block_delta",
          index: blockIndex,
          delta: { type: "input_json_delta", partial_json: args },
        });
      }
    }
  };
  const handleEvent = (eventText: string): void => {
    for (const line of eventText.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try {
        handleChunk(JSON.parse(data));
      } catch {
        // ignore a malformed chunk
      }
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // Normalise CRLF to LF so event boundaries ("\n\n") are found even
    // when the gateway frames SSE with \r\n line endings.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx = buffer.indexOf("\n\n");
    while (idx >= 0) {
      handleEvent(buffer.slice(0, idx));
      buffer = buffer.slice(idx + 2);
      idx = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleEvent(buffer);
  // A stream that yielded no content at all still needs one content block,
  // or the Messages wire has nothing to stop.
  if (openBlocks.length === 0) {
    const index = nextBlock++;
    startBlock(index, { type: "text", text: "" });
  }
  for (const index of openBlocks) {
    writeEvent("content_block_stop", { type: "content_block_stop", index });
  }
  const outTok =
    usage && typeof usage.completion_tokens === "number"
      ? Math.max(1, usage.completion_tokens)
      : Math.max(1, estimateOutputTokens(text));
  const inTok = usage && typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0;
  writeEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: mapAnthropicStop(finishReason || "stop") },
    usage: { input_tokens: inTok, output_tokens: outTok },
  });
  writeEvent("message_stop", { type: "message_stop" });
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

function readBody(req: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 8 * 1024 * 1024) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function failuresHeader(res: any, failures: FailureRow[]): void {
  if (failures.length === 0) return;
  // At most 8 rows and at most 8 KiB: a header over either limit is read
  // as no trail at all, so shrink (or drop) rather than overshoot.
  let rows = failures.slice(0, 8);
  let header = JSON.stringify(rows);
  while (rows.length > 1 && Buffer.byteLength(header) > 8192) {
    rows = rows.slice(0, -1);
    header = JSON.stringify(rows);
  }
  if (Buffer.byteLength(header) <= 8192) {
    res.setHeader("x-makefun-upstream-failures", header);
  }
}

function sendJson(res: any, status: number, obj: any, failures: FailureRow[]): void {
  failuresHeader(res, failures);
  const body = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) });
  res.end(body);
}

function sendError(res: any, wire: "openai" | "anthropic", status: number, fail: GatewayFail, failures: FailureRow[]): void {
  if (wire === "anthropic") {
    sendJson(res, status, { type: "error", error: { type: "api_error", message: fail.message } }, failures);
  } else {
    sendJson(
      res,
      status,
      { error: { message: fail.message, type: "gateway_error", code: fail.code } },
      failures,
    );
  }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

function handleModels(req: any, res: any, models: BriefModel[]): void {
  const ids = models.map((m) => m.id);
  const auth = String(req.headers["authorization"] || "");
  const apiKey = String(req.headers["x-api-key"] || "");
  if (apiKey && !auth) {
    // Anthropic shape.
    const trialNow = process.env.MAKEFUN_TRIAL_NOW || "2026-11-15T03:30:00Z";
    sendJson(
      res,
      200,
      {
        data: ids.map((id) => ({ id, type: "model", display_name: id, created_at: trialNow })),
        has_more: false,
        first_id: ids[0] || null,
        last_id: ids[ids.length - 1] || null,
      },
      [],
    );
  } else {
    // OpenAI shape (default, including Bearer auth).
    sendJson(
      res,
      200,
      {
        object: "list",
        data: ids.map((id) => ({ id, object: "model", created: 0, owned_by: "openrouter" })),
      },
      [],
    );
  }
}

async function handleOpenAI(req: any, res: any, body: any, models: BriefModel[]): Promise<void> {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const req2: Requirements = {
    json: !!(
      body.response_format &&
      (body.response_format.type === "json_object" || body.response_format.type === "json_schema")
    ),
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision: hasVision(messages),
  };
  const maxTokens = Math.min(
    8192,
    Math.max(
      1,
      typeof body.max_tokens === "number"
        ? body.max_tokens
        : typeof body.max_completion_tokens === "number"
          ? body.max_completion_tokens
          : 1024,
    ),
  );
  const candidates = rankCandidates(models, req2, estimateTokens(messages));
  const gatewayBody: any = {
    messages,
    max_tokens: maxTokens,
  };
  if (typeof body.temperature === "number") gatewayBody.temperature = body.temperature;
  if (typeof body.top_p === "number") gatewayBody.top_p = body.top_p;
  // Pass the client's response_format through whole: downgrading a
  // json_schema to json_object would drop the schema the client asked for.
  if (req2.json) gatewayBody.response_format = body.response_format;
  // An empty tools array is truthy in JS but must not reach the gateway;
  // tool_choice only makes sense beside real tools.
  if (req2.tools) {
    gatewayBody.tools = body.tools;
    if (body.tool_choice !== undefined) gatewayBody.tool_choice = body.tool_choice;
  }
  if (typeof body.stop === "string" || (Array.isArray(body.stop) && body.stop.length > 0)) {
    gatewayBody.stop = body.stop;
  }
  const stream = body.stream === true;
  if (stream) {
    // The wire check wants usage on a streamed answer; make sure the
    // gateway sends its usage chunk even if the client did not ask.
    gatewayBody.stream_options = {
      ...(body.stream_options && typeof body.stream_options === "object" ? body.stream_options : {}),
      include_usage: true,
    };
  }

  const result = await dialWithFailover(gatewayBody, candidates, stream);
  if (!result.ok) {
    sendError(res, "openai", result.last.status, result.last, result.failures);
    return;
  }
  const dialedId = result.model;
  if (!stream) {
    const gw = result.json || {};
    gw.model = dialedId;
    if (!gw.id) {
      openaiCounter++;
      gw.id = "chatcmpl-" + String(openaiCounter).padStart(6, "0");
    }
    if (!gw.usage) {
      const text = textOf((gw.choices && gw.choices[0] && gw.choices[0].message && gw.choices[0].message.content) || "");
      gw.usage = { prompt_tokens: 0, completion_tokens: estimateOutputTokens(text), total_tokens: estimateOutputTokens(text) };
    } else if (
      typeof gw.usage.completion_tokens === "number" &&
      gw.usage.completion_tokens < 1
    ) {
      // A real completion with text behind it must not report 0 output
      // tokens: the wire check needs an output count of at least 1.
      const text = textOf((gw.choices && gw.choices[0] && gw.choices[0].message && gw.choices[0].message.content) || "");
      if (text) {
        gw.usage.completion_tokens = estimateOutputTokens(text);
        if (typeof gw.usage.prompt_tokens === "number") {
          gw.usage.total_tokens = gw.usage.prompt_tokens + gw.usage.completion_tokens;
        }
      }
    }
    sendJson(res, 200, gw, result.failures);
    return;
  }
  failuresHeader(res, result.failures);
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  try {
    await forwardOpenAIStream(result.res, dialedId, (text) => res.write(text));
  } catch {
    // The gateway stream broke mid-way; the client sees a truncated stream.
  }
  res.end();
}

async function handleAnthropic(req: any, res: any, body: any, models: BriefModel[]): Promise<void> {
  const openaiMessages = anthropicMessagesToOpenAI(body);
  const hasTools = Array.isArray(body.tools) && body.tools.length > 0;
  const req2: Requirements = {
    json: false,
    tools: hasTools,
    vision: hasVision(openaiMessages),
  };
  const maxTokens = Math.min(
    8192,
    Math.max(1, typeof body.max_tokens === "number" ? body.max_tokens : 1024),
  );
  const candidates = rankCandidates(models, req2, estimateTokens(openaiMessages));
  const gatewayBody: any = {
    messages: openaiMessages,
    max_tokens: maxTokens,
  };
  if (typeof body.temperature === "number") gatewayBody.temperature = body.temperature;
  if (typeof body.top_p === "number") gatewayBody.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    gatewayBody.stop = body.stop_sequences;
  }
  const tools = anthropicToolsToOpenAI(body.tools);
  if (tools) gatewayBody.tools = tools;
  const toolChoice = anthropicToolChoiceToOpenAI(body.tool_choice);
  if (toolChoice !== undefined) gatewayBody.tool_choice = toolChoice;
  const stream = body.stream === true;

  const result = await dialWithFailover(gatewayBody, candidates, stream);
  if (!result.ok) {
    sendError(res, "anthropic", result.last.status, result.last, result.failures);
    return;
  }
  const dialedId = result.model;
  if (!stream) {
    const gw = result.json || {};
    gw.model = dialedId;
    sendJson(res, 200, openaiToAnthropic(gw, dialedId), result.failures);
    return;
  }
  failuresHeader(res, result.failures);
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  try {
    await forwardAnthropicStream(result.res, dialedId, (event, data) => {
      res.write("event: " + event + "\n" + "data: " + JSON.stringify(data) + "\n\n");
    });
  } catch {
    // The gateway stream broke mid-way; the client sees a truncated stream.
  }
  res.end();
}

// ---------------------------------------------------------------------------
// Env registration + work polling
// ---------------------------------------------------------------------------

async function registerAndPoll(envUrl: string, port: number, catalog: Catalog, server: any): Promise<void> {
  for (;;) {
    try {
      const r = await fetch(envUrl + "/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(10000),
      });
      if (r.ok) break;
    } catch {
      // env not up yet; retry
    }
    await sleep(1000);
  }
  for (;;) {
    await sleep(1000);
    try {
      const r = await fetch(envUrl + "/work", { signal: AbortSignal.timeout(10000) });
      if (!r.ok) continue;
      const j: any = await r.json().catch(() => null);
      if (j && j.done === true) {
        server.close(() => process.exit(0));
        setTimeout(() => process.exit(0), 2000).unref();
        return;
      }
    } catch {
      // transient; keep polling
    }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const brief = loadBrief();
const catalog = catalogFrom(brief.models);
const models = brief.models;

if (process.argv.slice(2).includes("--list")) {
  process.stdout.write(JSON.stringify(catalog) + "\n");
  process.exit(0);
}

const envUrl = (process.env.MAKEFUN_ENV_URL || "").replace(/\/+$/, "");

const server = http.createServer((req: any, res: any) => {
  (async () => {
    try {
      const url = new URL(req.url || "/", "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        handleModels(req, res, models);
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        const raw = await readBody(req);
        let body: any = {};
        try {
          body = JSON.parse(raw);
        } catch {
          sendJson(res, 400, { error: { message: "invalid JSON body", type: "invalid_request_error", code: "invalid_json" } }, []);
          return;
        }
        await handleOpenAI(req, res, body, models);
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/messages") {
        const raw = await readBody(req);
        let body: any = {};
        try {
          body = JSON.parse(raw);
        } catch {
          sendJson(res, 400, { type: "error", error: { type: "invalid_request_error", message: "invalid JSON body" } }, []);
          return;
        }
        await handleAnthropic(req, res, body, models);
        return;
      }
      sendJson(res, 404, { error: { message: "not found", type: "invalid_request_error", code: "not_found" } }, []);
    } catch (e: any) {
      if (!res.headersSent) {
        try {
          sendJson(res, 500, { error: { message: String((e && e.message) || e), type: "server_error", code: "internal_error" } }, []);
        } catch {
          // Socket already gone (e.g. oversized body destroyed the
          // request); there is no client left to answer, and throwing
          // here would crash the server on an unhandled rejection.
        }
      } else {
        try {
          res.end();
        } catch {
          // already closing
        }
      }
    }
  })();
});

server.listen(0, "0.0.0.0", () => {
  const addr: any = server.address();
  const port = addr && typeof addr.port === "number" ? addr.port : 0;
  if (envUrl) {
    registerAndPoll(envUrl, port, catalog, server);
  }
});
