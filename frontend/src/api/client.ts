import type {
  CancelResponse,
  ConfigDetail,
  ConfigJson,
  ConfigListResponse,
  ConfigSummary,
  DryRunRequest,
  DryRunResponse,
  EnvStatus,
  RunDetailResponse,
  RunListItem,
  RunStartRequest,
  RunStartResponse,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${path}`;
  const headers: HeadersInit = {
    Accept: "application/json",
    ...(init.headers || {}),
  };
  const response = await fetch(url, { ...init, headers });

  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const detail =
      (isJson && (payload?.detail || payload?.message)) ||
      (typeof payload === "string" ? payload : JSON.stringify(payload));
    throw new Error(detail || `Request failed: ${response.status}`);
  }

  return payload as T;
}

function encodePath(path: string): string {
  return encodeURIComponent(path);
}

export async function getHealth(): Promise<{ status: string }> {
  return request("/health");
}

export async function getEnvStatus(): Promise<EnvStatus> {
  return request("/env/status");
}

export async function listConfigs(): Promise<ConfigListResponse> {
  return request("/configs");
}

export async function getConfigDetail(path: string): Promise<ConfigDetail> {
  return request(`/configs/${encodePath(path)}`);
}

export async function getConfigSummary(path: string): Promise<ConfigSummary> {
  return request(`/configs/${encodePath(path)}/summary`);
}

export async function getConfigAsJson(path: string): Promise<ConfigJson> {
  return request(`/configs/${encodePath(path)}/as-json`);
}

export async function postDryRun(payload: DryRunRequest): Promise<DryRunResponse> {
  return request("/runs/dry-run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function postRun(payload: RunStartRequest): Promise<RunStartResponse> {
  return request("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function listRuns(status?: string): Promise<RunListItem[]> {
  const suffix = status ? `?status_filter=${encodeURIComponent(status)}` : "";
  return request(`/runs${suffix}`);
}

export async function getRunDetail(runId: string): Promise<RunDetailResponse> {
  return request(`/runs/${encodePath(runId)}`);
}

export async function cancelRun(runId: string): Promise<CancelResponse> {
  return request(`/runs/${encodePath(runId)}/cancel`, { method: "POST" });
}
