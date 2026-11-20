// Shapes of the server's JSON (GET /v1/config and friends) and of the UI's own state. Every field the
// server may leave out is optional: the UI carries no defaults of its own.

export interface Range {
  min: number;
  max: number;
  step: number;
}

/** Valid ranges by setting name, plus the one plain number the server reports (max_stop). */
export interface Limits {
  max_stop?: number;
  [name: string]: Range | number | undefined;
}

/** The ranges of `limits` (everything except the plain numbers). */
export type Ranges = Record<string, Range | undefined>;

export interface ConfigDefaults {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  max_tokens?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  penalty_last_n?: number;
  seed?: number;
  json_mode?: boolean;
  ignore_eos?: boolean;
  cache_prompt?: boolean;
  best_of?: number;
  mcts?: {
    simulations?: number;
    expand?: number;
    max_depth?: number;
    step_tokens?: number;
    c?: number;
    temperature?: number;
  };
  system_prompt?: string;
  stop?: string[];
  watermark?: { enabled?: boolean; gamma?: number; delta?: number; server_key?: boolean };
}

export interface ParallelInfo {
  requested: number;
  slots: number;
  max: number;
  batch_tokens: number;
  batch_tokens_max: number;
  note?: string;
}

export interface Features {
  prompt_cache?: boolean;
  best_of?: boolean;
  mcts?: boolean;
  request_chat_template?: boolean;
  system_prompt?: boolean;
  parallel?: ParallelInfo;
  [name: string]: unknown;
}

export interface ConfigInfo {
  keys?: Record<string, unknown>;
  sources?: Record<string, string>;
  path?: string;
  file_used?: boolean;
  writable?: boolean;
}

export interface Runtime {
  act_quant?: { requested: string; effective?: string };
  quality_tolerance?: number;
  classifier?: string;
  self_check_tokens?: number;
  attention_kernel?: string;
  kv_type?: string;
  kv_offload?: boolean;
  simd?: string;
  threads?: number;
  distributed?: { layers: string; host: string; port: number }[];
}

export interface ModelInfo {
  id?: string;
  owned_by?: string;
  chat_format?: string;
  context_length?: number;
  dim?: number;
  n_layers?: number;
  moe?: boolean;
  recurrent?: boolean;
  vision?: boolean;
}

export interface ChatTemplateInfo {
  text?: string;
  source?: string;
  max_bytes?: number;
}

export interface ServerConfig {
  defaults?: ConfigDefaults;
  limits?: Limits;
  features?: Features;
  chat_template?: ChatTemplateInfo;
  config?: ConfigInfo;
  runtime?: Runtime;
  model?: ModelInfo;
  ui?: { title?: string; logo?: string; logo_ascii?: string };
  [section: string]: unknown;
}

/** The generation settings the Settings panel edits; every value starts from the server's defaults. */
export interface Settings {
  temperature: number | undefined;
  top_p: number | undefined;
  top_k: number | undefined;
  min_p: number | undefined;
  max_tokens: number | undefined;
  repetition_penalty: number | undefined;
  presence_penalty: number | undefined;
  frequency_penalty: number | undefined;
  penalty_last_n: number | undefined;
  seed: number;
  json_mode: boolean;
  ignore_eos: boolean;
  cache_prompt: boolean;
  best_of: number;
  mcts_simulations: number;
  mcts_expand: number | undefined;
  mcts_max_depth: number | undefined;
  mcts_step_tokens: number | undefined;
  mcts_c: number | undefined;
  mcts_temperature: number | undefined;
  system_prompt: string;
  chat_template: string;
  stop: string;
  watermark_enabled: boolean;
  watermark_key: string;
  watermark_gamma: number | undefined;
  watermark_delta: number | undefined;
}

export type SettingKey = keyof Settings;
export type SettingValue = Settings[SettingKey];

export interface ChatMessage {
  role: string;
  content: string;
}

export interface UpdateInfo {
  status?: string;
  current?: string;
  latest?: string;
  prerelease?: boolean;
  url?: string;
  checking?: boolean;
  checked_at?: number;
  settings?: { enabled?: boolean; [field: string]: unknown };
  [field: string]: unknown;
}

export interface WatermarkResult {
  watermarked?: boolean;
  hit?: boolean;
  green_tokens?: number;
  tokens_scored?: number;
  z_score: number;
  z_threshold?: number;
}

export interface SelfCheckResult {
  verdict: string;
  act_quant: string;
  classifier: string;
  delta_pct: number;
  tolerance_pct: number;
  perplexity: { reference: number; configured: number };
  note: string;
}

export interface PromptCacheStats {
  budget_bytes: number;
  bytes_used: number;
  entries: number;
  hits: number;
}

export interface SaveResult {
  saved: boolean;
  applies: string;
  overridden_by: string | null;
  path: string;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

export interface StreamSummary {
  finish_reason?: string;
  usage?: Usage;
  best_of?: unknown;
}

export interface StatusUpdate {
  queued: number;
}
