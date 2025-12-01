export type Dict = Record<string, unknown>;

export interface ConfigItem {
  name: string;
  path: string;
  tags: string[];
}

export interface ConfigListResponse {
  root: string;
  items: ConfigItem[];
}

export interface ConfigDetail {
  name: string;
  path: string;
  yaml: string;
}

export interface ConfigJson {
  path: string;
  config: Dict;
}

export interface ConfigSummary {
  path: string;
  name: string;
  summary: Dict;
}

export interface ValidationProblem {
  path: string[];
  message: string;
}

export interface EnvStatus {
  ok: boolean;
  missing_keys: string[];
  providers: Record<
    string,
    {
      ok: boolean;
      missing: string[];
      present: { name: string; masked: string }[];
    }
  >;
}

export interface DryRunRequest {
  config_path: string;
  run_id?: string;
  options?: RunOptionsInput;
  ping_llm?: boolean;
}

export interface RunOptionsInput {
  max_trials?: number;
  sampler?: string | null;
  llm_enabled?: boolean | null;
  seed?: number | null;
}

export interface DryRunResponse {
  run_id: string;
  config_path: string;
  config: Dict;
}

export interface RunStartRequest extends DryRunRequest {
  perform_dry_run?: boolean;
  llm_comparison?: boolean;
}

export interface RunHandle {
  run_id: string;
  status: string;
  run_dir: string;
  meta_path: string;
}

export interface RunComparisonInfo {
  comparison_id: string;
  shared_seed?: number | null;
  record_path?: string | null;
  llm_enabled: RunHandle;
  llm_disabled: RunHandle;
}

export interface RunStartResponse {
  run_id: string;
  status: string;
  run_dir: string;
  meta_path: string;
  comparison?: RunComparisonInfo | null;
}

export interface RunListItem {
  run_id: string;
  status?: string | null;
  created_at?: string | null;
  metadata: Dict;
  report: Dict;
  artifacts: Dict;
  status_payload: Dict;
  run_dir?: string | null;
}

export interface JobInfo {
  pid?: number | null;
  state?: string | null;
  cancel_requested?: boolean | null;
  started_at?: string | null;
}

export interface RunDetailResponse {
  meta: Dict;
  config?: Dict | null;
  job?: JobInfo | null;
}

export interface CancelResponse {
  ok: boolean;
  message: string;
}
