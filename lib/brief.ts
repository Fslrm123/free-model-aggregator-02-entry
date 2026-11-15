// The brief: the free model list and the six prompts the trial hands the entry.
//
// The brief file is plain text holding two JSON documents: the model list (its
// `models` or `ratings` array) and the prompt manifest (its `prompts` or
// `manifest` array). This module finds those documents, reads the rows, builds
// the one catalog the entry registers, and orders the models a prompt may dial
// by the house's published rating — best first, so the first call is usually
// the only call the prompt costs.
//
// Nothing here reads the wall clock or any randomness: the same brief gives the
// same order every run, which is what a replay needs.

import { readFileSync } from 'node:fs';

/** Any JSON object. */
export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One free model row, as the brief spells it. */
export interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  rating: number;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What a prompt needs its model to do. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One prompt row, in the order the env sends them. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model of the catalog document the entry registers and `--list` prints. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** The catalog document, byte for byte the one the register body carries. */
export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
}

const SCHEMA_MARK = 'free-model-aggregator-02';
const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolValue(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/**
 * The top-level JSON objects in `text`, found by brace matching with string
 * contents skipped, in the order they appear.
 */
function topObjects(text: string): Json[] {
  const found: Json[] = [];
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf('{', i);
    if (start === -1) break;
    let depth = 0;
    let inString = false;
    let j = start;
    for (; j < text.length; j++) {
      const ch = text[j];
      if (inString) {
        if (ch === '\\') j += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(start, j + 1));
      if (isRecord(parsed)) found.push(parsed);
    } catch {
      // Not a JSON object: keep scanning past it.
    }
    i = j + 1;
  }
  return found;
}

/** The first document under this contest's schema that carries one of `keys`. */
function documentWith(docs: readonly Json[], keys: readonly string[]): Json | null {
  for (const doc of docs) {
    if (!textValue(doc.schema).includes(SCHEMA_MARK)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return doc;
  }
  return null;
}

/** One model row; a bare vendor id under an openrouter provider is routed with the prefix. */
function modelRow(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const raw = textValue(value.id) !== '' ? textValue(value.id) : textValue(value.model);
  if (raw === '') return null;
  const named = textValue(value.provider) !== '' ? textValue(value.provider) : textValue(value.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    displayName: textValue(value.display_name) !== '' ? textValue(value.display_name) : raw,
    rating: finiteNumber(value.rating_milli) ?? 0,
    contextLength: finiteNumber(value.context_length),
    tools: boolValue(value.tools),
    vision: boolValue(value.vision),
    json: boolValue(value.json),
    free: value.free !== false,
  };
}

function promptRow(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = textValue(value.id);
  if (id === '') return null;
  const req = isRecord(value.requires) ? value.requires : {};
  const requires: Requirement = {};
  if (req.tools === true) requires.tools = true;
  if (req.vision === true) requires.vision = true;
  if (req.json === true) requires.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) requires.minContext = req.min_context;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires,
  };
}

/** Read the brief at `path`; an empty brief when there is no file to read. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = topObjects(text);
  const modelsDoc = documentWith(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief at ${path} holds no models document` };
  const modelsKey = Array.isArray(modelsDoc.models) ? 'models' : 'ratings';
  const models = (modelsDoc[modelsKey] as unknown[]).flatMap((v) => {
    const row = modelRow(v);
    return row === null ? [] : [row];
  });
  const promptsDoc = documentWith(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief at ${path} holds no prompts document` };
  const promptsKey = Array.isArray(promptsDoc.prompts) ? 'prompts' : 'manifest';
  const prompts = (promptsDoc[promptsKey] as unknown[]).flatMap((v) => {
    const row = promptRow(v);
    return row === null ? [] : [row];
  });
  return { models, prompts, error: null };
}

/** The brief's free models, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model's capabilities meet a prompt's requirements; unknown never meets one. */
export function meets(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.minContext !== undefined && (model.contextLength === null || model.contextLength < requires.minContext)) return false;
  return true;
}

/**
 * The free models that can serve `requires`, best first: the house rating,
 * then the wider context window, then the id, so the order is fixed for a
 * given brief and two runs never disagree about who goes first.
 */
export function candidatesFor(brief: Brief, requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((m) => meets(requires, m))
    .sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      const ac = a.contextLength ?? -1;
      const bc = b.contextLength ?? -1;
      if (bc !== ac) return bc - ac;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
}

/**
 * The catalog the brief's free models make: one openrouter provider, the
 * fourteen dialable ids exactly, no more and no fewer. The base_url is
 * OpenRouter's public endpoint, never the gateway's trial address, because the
 * catalog is part of the state export a replay must reproduce byte for byte.
 */
export function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of freeModels(brief)) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
