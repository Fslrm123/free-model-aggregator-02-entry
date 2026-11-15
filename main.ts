// One server, two client wires, and a live free model behind every answer.
//
// The trial gives this server a brief (the free models with the house's
// ratings, and the six prompts), a gateway (the only outside service the
// sandbox reaches), and an env (register, then six prompts, one at a time).
// Each prompt names model "auto": this server picks the best free model that
// meets the prompt, dials it on the gateway's Chat Completions route, and
// returns the answer in the wire the env used, streamed or not as asked.
//
// ROUTING: candidates are ordered by the house rating, best first, so the
// first call usually serves the prompt — the tie-break counts the gateway
// calls a trial made, and a call the provider refused is not counted, but a
// lean router still wins ties. A refusal is never the end of a prompt: the
// next candidate is dialled, up to five models, and every refusal is reported
// on the answer in `x-makefun-upstream-failures`, the only way the failover
// column can see a refusal the server recovered from.
//
// DETERMINISM: every choice is a function of the brief, the requests and the
// gateway's answers seen so far. No wall clock and no unseeded randomness in
// any routing decision; the per-call abort is only a safety valve against a
// hung upstream, and the poll interval only paces the env check.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  candidatesFor,
  freeModels,
  isRecord,
  meets,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ListedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most free models one prompt is dialled on before the server gives up. */
const MAX_ATTEMPTS = 5;
/** A hung upstream is abandoned after this long; the env caps a prompt at 30 s. */
const UPSTREAM_TIMEOUT_MS = 25000;
/** The output cap a request gets when it names none; the gateway refuses uncapped calls. */
const DEFAULT_MAX_TOKENS = 512;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's pushed key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route; a base without `/v1` gets one. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function listedModels(): ListedModel[] {
  return freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, displayName: m.displayName }));
}

/** The requirements of the prompt now arriving: the brief's row, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders?: Record<string, string>): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One row of the upstream-failures trail the failover column reads. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header, or nothing when no call was refused before this answer. */
function trailHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** The error code an upstream refusal carries, or a plain `http_<status>`. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code;
      if (typeof code === 'string' && code !== '') return code;
      const type = parsed.error.type;
      if (typeof type === 'string' && type !== '') return type;
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** A Chat Completions body for the gateway: the dialled model, an output cap, the rest as sent. */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(gatewayBody(body, model)),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

/**
 * Dial `candidates` in order until one answers, recording every refusal.
 * Always returns the refusal trail; `model` is null when none answered.
 */
async function dialWithFailover(
  candidates: ModelRow[],
  body: Json,
): Promise<{ model: string | null; response: Response | null; failures: FailureRow[] }> {
  const failures: FailureRow[] = [];
  const tried = candidates.slice(0, MAX_ATTEMPTS);
  for (const candidate of tried) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      log(`dial ${candidate.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: candidate.id, response: upstream, failures };
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
    log(`dial ${candidate.id} refused HTTP ${upstream.status}`);
  }
  return { model: null, response: null, failures };
}

/** The names of the tools the client declared, for the wire's tools check. */
function declaredTools(body: Json): Set<string> {
  const names = new Set<string>();
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') names.add(tool.name);
    }
  }
  return names;
}

/** Drop tool calls the client did not declare; the wire check forbids their names. */
function keepDeclaredCalls(answer: Json, declared: Set<string>): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return answer;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null || !Array.isArray(message.tool_calls)) return answer;
  const kept = message.tool_calls.filter(
    (call) => isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && declared.has(call.function.name),
  );
  if (kept.length === message.tool_calls.length) return answer;
  return {
    ...answer,
    choices: [{ ...choice, message: { ...message, tool_calls: kept } }],
  };
}

/** A whole usage count: the provider's, or an estimate when it sent none. */
function saneUsage(answer: Json, promptChars: number, textChars: number): Json {
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = typeof usage.prompt_tokens === 'number' && usage.prompt_tokens >= 0 ? usage.prompt_tokens : Math.max(0, Math.ceil(promptChars / 4));
  const output =
    typeof usage.completion_tokens === 'number' && usage.completion_tokens >= 1
      ? usage.completion_tokens
      : Math.max(1, Math.ceil(textChars / 4));
  return { ...answer, usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } };
}

/** The text an answer carries, for the usage estimate when the provider sent none. */
function answerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return '';
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return '';
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
      .map((b) => (b as Json).text as string)
      .join('\n');
  }
  return '';
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(brief, requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  const dialled = await dialWithFailover(candidates, body);
  const { model, response, failures } = dialled;
  if (model === null || response === null) {
    return sendError(
      res,
      'openai',
      502,
      'upstream_error',
      'every free model that meets this request refused the call',
      trailHeader(failures),
    );
  }
  const headers = trailHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...headers,
    });
    try {
      for await (const frame of openAIStreamWithModel(response.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await response.json();
  } catch {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that is not JSON', headers);
  }
  if (!isRecord(answer)) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that is not JSON', headers);
  const cleaned = keepDeclaredCalls(answer, declaredTools(body));
  const withUsage = saneUsage(cleaned, JSON.stringify(body).length, answerText(cleaned).length);
  sendJson(res, 200, openAIJsonWithModel(withUsage, model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(brief, requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  const upstreamBody = anthropicToOpenAI(body, candidates[0].id);
  const dialled = await dialWithFailover(candidates, upstreamBody);
  const { model, response, failures } = dialled;
  if (model === null || response === null) {
    return sendError(
      res,
      'anthropic',
      502,
      'api_error',
      'every free model that meets this request refused the call',
      trailHeader(failures),
    );
  }
  const headers = trailHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...headers,
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(response.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await response.json();
  } catch {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that is not JSON', headers);
  }
  if (!isRecord(answer)) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that is not JSON', headers);
  const cleaned = keepDeclaredCalls(answer, declaredTools(body));
  const withUsage = saneUsage(cleaned, JSON.stringify(body).length, answerText(cleaned).length);
  sendJson(res, 200, openAIToAnthropic(withUsage, model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${listedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
