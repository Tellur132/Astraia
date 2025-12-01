import { useEffect, useMemo, useState } from "react";
import {
  cancelRun,
  getConfigAsJson,
  getConfigDetail,
  getConfigSummary,
  getEnvStatus,
  getHealth,
  getRunDetail,
  listConfigs,
  listRuns,
  postDryRun,
  postRun,
} from "./api/client";
import type {
  ConfigItem,
  ConfigListResponse,
  DryRunResponse,
  EnvStatus,
  RunDetailResponse,
  RunListItem,
  RunOptionsInput,
} from "./api/types";

type ConfigTab = "summary" | "json" | "yaml";
type LlmChoice = "default" | "enabled" | "disabled";

interface RunFormState {
  configPath: string;
  runId: string;
  maxTrials: string;
  sampler: string;
  seed: string;
  llmChoice: LlmChoice;
  performDryRun: boolean;
  pingLlm: boolean;
  llmComparison: boolean;
}

const defaultRunForm: RunFormState = {
  configPath: "",
  runId: "",
  maxTrials: "",
  sampler: "",
  seed: "",
  llmChoice: "default",
  performDryRun: true,
  pingLlm: true,
  llmComparison: false,
};

function App() {
  const [health, setHealth] = useState<string>("unknown");
  const [envStatus, setEnvStatus] = useState<EnvStatus | null>(null);
  const [configRoot, setConfigRoot] = useState<string>("");
  const [configs, setConfigs] = useState<ConfigItem[]>([]);
  const [configFilter, setConfigFilter] = useState("");
  const [configLoading, setConfigLoading] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<ConfigItem | null>(null);
  const [configSummary, setConfigSummary] = useState<Record<string, unknown> | null>(null);
  const [configJson, setConfigJson] = useState<Record<string, unknown> | null>(null);
  const [configYaml, setConfigYaml] = useState<string>("");
  const [configTab, setConfigTab] = useState<ConfigTab>("summary");

  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runFilter, setRunFilter] = useState<"all" | "running" | "completed" | "failed">("all");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetailResponse | null>(null);

  const [runForm, setRunForm] = useState<RunFormState>(defaultRunForm);
  const [dryRunResult, setDryRunResult] = useState<DryRunResponse | null>(null);
  const [dryRunLoading, setDryRunLoading] = useState(false);
  const [launchLoading, setLaunchLoading] = useState(false);

  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    refreshBasics();
    refreshConfigs();
    refreshRuns(runFilter);
  }, []);

  useEffect(() => {
    refreshRuns(runFilter);
  }, [runFilter]);

  const filteredConfigs = useMemo(() => {
    const keyword = configFilter.trim().toLowerCase();
    if (!keyword) {
      return configs;
    }
    return configs.filter((item) => {
      const haystack = [item.name, item.path, ...(item.tags || [])]
        .join(" ")
        .toLowerCase();
      return haystack.includes(keyword);
    });
  }, [configs, configFilter]);

  const selectedRun = useMemo(
    () => runs.find((r) => r.run_id === selectedRunId) || null,
    [runs, selectedRunId],
  );

  async function refreshBasics() {
    try {
      const [healthRes, envRes] = await Promise.all([getHealth(), getEnvStatus()]);
      setHealth(healthRes.status || "unknown");
      setEnvStatus(envRes);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function refreshConfigs() {
    try {
      const res: ConfigListResponse = await listConfigs();
      setConfigRoot(res.root);
      setConfigs(res.items);
      if (!selectedConfig && res.items.length > 0) {
        handleSelectConfig(res.items[0]);
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function handleSelectConfig(item: ConfigItem) {
    setSelectedConfig(item);
    setRunForm((prev) => ({ ...prev, configPath: item.path }));
    setConfigLoading(true);
    setMessage("");
    setError("");
    try {
      const [detail, summaryRes, jsonRes] = await Promise.all([
        getConfigDetail(item.path),
        getConfigSummary(item.path),
        getConfigAsJson(item.path),
      ]);
      setConfigYaml(detail.yaml);
      setConfigSummary(summaryRes.summary);
      setConfigJson(jsonRes.config);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setConfigLoading(false);
    }
  }

  async function refreshRuns(filter: typeof runFilter) {
    setRunsLoading(true);
    try {
      const data = await listRuns(filter === "all" ? undefined : filter);
      setRuns(data);
      if (selectedRunId) {
        const stillExists = data.some((entry) => entry.run_id === selectedRunId);
        if (stillExists) {
          loadRunDetail(selectedRunId);
        } else {
          setSelectedRunId(null);
          setRunDetail(null);
        }
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunsLoading(false);
    }
  }

  async function loadRunDetail(runId: string) {
    try {
      const detail = await getRunDetail(runId);
      setRunDetail(detail);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function openRun(runId: string) {
    setSelectedRunId(runId);
    await loadRunDetail(runId);
  }

  function buildOptions(): RunOptionsInput | undefined {
    const options: RunOptionsInput = {};
    const max = parseInt(runForm.maxTrials, 10);
    if (!Number.isNaN(max) && max > 0) {
      options.max_trials = max;
    }
    const seed = parseInt(runForm.seed, 10);
    if (!Number.isNaN(seed)) {
      options.seed = seed;
    }
    if (runForm.sampler.trim()) {
      options.sampler = runForm.sampler.trim();
    }
    if (runForm.llmChoice !== "default") {
      options.llm_enabled = runForm.llmChoice === "enabled";
    }
    return Object.keys(options).length ? options : undefined;
  }

  async function handleDryRun() {
    if (!runForm.configPath) {
      setError("Config path を選択してください。");
      return;
    }
    setDryRunLoading(true);
    setMessage("");
    setError("");
    try {
      const res = await postDryRun({
        config_path: runForm.configPath,
        run_id: runForm.runId || undefined,
        options: buildOptions(),
        ping_llm: runForm.pingLlm,
      });
      setDryRunResult(res);
      setMessage(`Dry-run OK: run_id=${res.run_id}`);
      setRunForm((prev) => ({ ...prev, runId: prev.runId || res.run_id }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDryRunLoading(false);
    }
  }

  async function handleLaunch() {
    if (!runForm.configPath) {
      setError("Config path を選択してください。");
      return;
    }
    setLaunchLoading(true);
    setMessage("");
    setError("");
    try {
      const res = await postRun({
        config_path: runForm.configPath,
        run_id: runForm.runId || undefined,
        options: buildOptions(),
        perform_dry_run: runForm.performDryRun,
        ping_llm: runForm.pingLlm,
        llm_comparison: runForm.llmComparison,
      });
      if (res.comparison) {
        setMessage(
          `LLM比較を開始: LLM=${res.run_id} / baseline=${res.comparison.llm_disabled.run_id} (seed=${res.comparison.shared_seed ?? "auto"})`,
        );
      } else {
        setMessage(`Run started: ${res.run_id}`);
      }
      setSelectedRunId(res.run_id);
      await refreshRuns(runFilter);
      await loadRunDetail(res.run_id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLaunchLoading(false);
    }
  }

  async function handleCancel(runId: string) {
    try {
      await cancelRun(runId);
      setMessage(`Run ${runId} にキャンセルを送信しました。`);
      await refreshRuns(runFilter);
      await loadRunDetail(runId);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="app-shell">
      <div className="headline">
        <div>
          <h1>Astraia Control Room</h1>
          <div className="muted">設定のブラウズから実行・監視までをまとめて管理</div>
        </div>
        <span className="badge">Local first · FastAPI + React</span>
      </div>

      <div className="status-cards">
        <div className="status-card">
          <div className="label">Backend</div>
          <div className="value">
            <StatusPill status={health === "ok" ? "ok" : "bad"} text={health} />
          </div>
        </div>
        <div className="status-card">
          <div className="label">Env Keys</div>
          <div className="value">
            <StatusPill status={envStatus?.ok ? "ok" : "warn"} text={envStatus?.ok ? "Ready" : "Check"} />
          </div>
          {envStatus && !envStatus.ok ? (
            <div className="small">Missing: {envStatus.missing_keys.join(", ") || "-"}</div>
          ) : null}
        </div>
        <div className="status-card">
          <div className="label">Config root</div>
          <div className="value small">{configRoot || "未取得"}</div>
        </div>
        <div className="status-card">
          <div className="label">Runs</div>
          <div className="value">
            <strong>{runs.length}</strong>
            <span className="muted">tracked</span>
          </div>
        </div>
      </div>

      <div className="grid">
        <section className="panel">
          <div className="section-header">
            <h2>Config ブラウザ</h2>
            <div className="actions">
              <input
                type="text"
                placeholder="検索 (名前 / パス / タグ)"
                value={configFilter}
                onChange={(e) => setConfigFilter(e.target.value)}
              />
              <button className="secondary" onClick={refreshConfigs}>
                再読込
              </button>
            </div>
          </div>

          <div className="config-list">
            <div className="config-items">
              {filteredConfigs.map((item) => (
                <div
                  key={item.path}
                  className={`config-item ${selectedConfig?.path === item.path ? "active" : ""}`}
                  onClick={() => handleSelectConfig(item)}
                >
                  <div style={{ fontWeight: 700 }}>{item.name}</div>
                  <div className="small">{item.path}</div>
                  <div>
                    {item.tags?.map((tag) => (
                      <span key={tag} className="tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
              {!filteredConfigs.length && <div className="muted">Config が見つかりませんでした。</div>}
            </div>

            <div className="config-detail">
              <div className="tabs">
                {(["summary", "json", "yaml"] as ConfigTab[]).map((tab) => (
                  <div
                    key={tab}
                    className={`tab ${configTab === tab ? "active" : ""}`}
                    onClick={() => setConfigTab(tab)}
                  >
                    {tab.toUpperCase()}
                  </div>
                ))}
              </div>
              {configLoading && <div className="muted">読み込み中...</div>}
              {!configLoading && configTab === "summary" && (
                <div className="code-block">
                  {configSummary ? <pre>{JSON.stringify(configSummary, null, 2)}</pre> : <div className="muted">サマリ未取得</div>}
                </div>
              )}
              {!configLoading && configTab === "json" && (
                <div className="code-block">
                  {configJson ? <pre>{JSON.stringify(configJson, null, 2)}</pre> : <div className="muted">JSON 未取得</div>}
                </div>
              )}
              {!configLoading && configTab === "yaml" && (
                <div className="code-block">
                  {configYaml ? <pre>{configYaml}</pre> : <div className="muted">YAML 未取得</div>}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="section-header">
            <h2>Run 起動 & 管理</h2>
            <div className="actions">
              <select value={runFilter} onChange={(e) => setRunFilter(e.target.value as typeof runFilter)}>
                <option value="all">All</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
              <button className="secondary" onClick={() => refreshRuns(runFilter)}>
                更新
              </button>
            </div>
          </div>

          <div className="panel" style={{ marginBottom: 12 }}>
            <h3>実行フォーム</h3>
            <div className="run-form">
              <div className="full">
                <label>
                  Config path
                  <select
                    value={runForm.configPath}
                    onChange={(e) => setRunForm((prev) => ({ ...prev, configPath: e.target.value }))}
                  >
                    <option value="">選択してください</option>
                    {configs.map((cfg) => (
                      <option key={cfg.path} value={cfg.path}>
                        {cfg.name} ({cfg.path})
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <label>
                run_id (任意)
                <input
                  value={runForm.runId}
                  onChange={(e) => setRunForm((prev) => ({ ...prev, runId: e.target.value }))}
                  placeholder="未指定なら自動生成"
                />
              </label>
              <label>
                max_trials override
                <input
                  type="number"
                  min="1"
                  value={runForm.maxTrials}
                  onChange={(e) => setRunForm((prev) => ({ ...prev, maxTrials: e.target.value }))}
                  placeholder="例: 20"
                />
              </label>
              <label>
                Sampler override
                <input
                  value={runForm.sampler}
                  onChange={(e) => setRunForm((prev) => ({ ...prev, sampler: e.target.value }))}
                  placeholder="nsga2, tpe など"
                />
              </label>
              <label>
                seed (任意)
                <input
                  type="number"
                  min="0"
                  value={runForm.seed}
                  onChange={(e) => setRunForm((prev) => ({ ...prev, seed: e.target.value }))}
                  placeholder="固定する場合は整数を指定"
                />
              </label>
              <label>
                LLM
                <select
                  value={runForm.llmChoice}
                  onChange={(e) => setRunForm((prev) => ({ ...prev, llmChoice: e.target.value as LlmChoice }))}
                >
                  <option value="default">設定通り</option>
                  <option value="enabled">強制有効</option>
                  <option value="disabled">強制無効</option>
                </select>
              </label>
              <label className="full">
                オプション
                <div className="split">
                  <label>
                    <input
                      type="checkbox"
                      checked={runForm.performDryRun}
                      onChange={(e) => setRunForm((prev) => ({ ...prev, performDryRun: e.target.checked }))}
                    />{" "}
                    実行前に dry-run
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={runForm.pingLlm}
                      onChange={(e) => setRunForm((prev) => ({ ...prev, pingLlm: e.target.checked }))}
                    />{" "}
                    LLM ping を行う
                  </label>
                </div>
              </label>
              <label className="full">
                <input
                  type="checkbox"
                  checked={runForm.llmComparison}
                  onChange={(e) => setRunForm((prev) => ({ ...prev, llmComparison: e.target.checked }))}
                />{" "}
                LLM比較モード（LLM無効のベースラインも同時実行し比較用レコードを生成）
              </label>
            </div>
            <div className="actions" style={{ marginTop: 10 }}>
              <button className="secondary" onClick={handleDryRun} disabled={dryRunLoading}>
                {dryRunLoading ? "実行中..." : "Dry-run"}
              </button>
              <button onClick={handleLaunch} disabled={launchLoading}>
                {launchLoading ? "起動中..." : "Run を開始"}
              </button>
              <button className="secondary" onClick={() => setRunForm(defaultRunForm)}>
                リセット
              </button>
            </div>
            {dryRunResult && (
              <div className="muted" style={{ marginTop: 8 }}>
                Dry-run: run_id={dryRunResult.run_id} / config={dryRunResult.config_path}
              </div>
            )}
          </div>

          <div className="panel">
            <h3>Runs</h3>
            {runsLoading && <div className="muted">読み込み中...</div>}
            {!runsLoading && (
              <div className="run-list">
                {runs.map((run) => (
                  <div key={run.run_id} className="run-card">
                    <div className="header">
                      <div>
                        <div style={{ fontWeight: 700 }}>{run.run_id}</div>
                        <div className="small">{run.created_at || "created_at n/a"}</div>
                      </div>
                      <div className="actions">
                        <StatusPill status={run.status || "unknown"} text={run.status || "unknown"} />
                        <button className="secondary" onClick={() => openRun(run.run_id)}>
                          詳細
                        </button>
                        {(run.status === "running" || run.status === "cancelling") && (
                          <button className="secondary" onClick={() => handleCancel(run.run_id)}>
                            キャンセル
                          </button>
                        )}
                      </div>
                    </div>
                    <div className="small">
                      config: {String((run.metadata?.config_path as string) || run.metadata?.name || "n/a")}
                    </div>
                    {(() => {
                      const note = run.status_payload?.note as unknown;
                      return note !== undefined && note !== null ? (
                        <div className="small">note: {String(note)}</div>
                      ) : null;
                    })()}
                  </div>
                ))}
                {!runs.length && <div className="muted">まだ run はありません。</div>}
              </div>
            )}
          </div>

          {selectedRun && runDetail && (
            <div className="panel" style={{ marginTop: 12 }}>
              <h3>Run detail: {selectedRun.run_id}</h3>
              <div className="split">
                <div className="status-card" style={{ flex: 1 }}>
                  <div className="label">Status</div>
                  <div className="value">
                    <StatusPill status={selectedRun.status || "unknown"} text={selectedRun.status || "unknown"} />
                  </div>
                  {runDetail.job?.state && <div className="small">Job: {runDetail.job.state}</div>}
                  {runDetail.job?.pid && <div className="small">PID: {runDetail.job.pid}</div>}
                  {runDetail.job?.cancel_requested && <div className="small">cancel requested</div>}
                </div>
                <div className="status-card" style={{ flex: 1 }}>
                  <div className="label">Artifacts</div>
                  <div className="small">{selectedRun.run_dir || "-"}</div>
                  {selectedRun.artifacts?.report && (
                    <div className="small">report: {String(selectedRun.artifacts.report)}</div>
                  )}
                </div>
              </div>
              <div className="code-block" style={{ marginTop: 10 }}>
                <pre>{JSON.stringify(runDetail.meta, null, 2)}</pre>
              </div>
              {runDetail.config && (
                <div className="code-block" style={{ marginTop: 10 }}>
                  <pre>{JSON.stringify(runDetail.config, null, 2)}</pre>
                </div>
              )}
            </div>
          )}
        </section>
      </div>

      {(message || error) && (
        <div className="panel" style={{ marginTop: 16, borderColor: error ? "rgba(255,107,129,0.45)" : undefined }}>
          {message && <div style={{ color: "#9ff4d6" }}>{message}</div>}
          {error && <div style={{ color: "#ffc0cb" }}>{error}</div>}
        </div>
      )}
    </div>
  );
}

function StatusPill({ status, text }: { status: string; text: string }) {
  const tone = statusTone(status);
  return <span className={`pill ${tone}`}>{text}</span>;
}

function statusTone(status: string) {
  if (!status) return "warn";
  const normalized = status.toLowerCase();
  if (["ok", "ready", "completed", "success", "running"].includes(normalized)) {
    return "ok";
  }
  if (["failed", "error", "bad"].includes(normalized)) {
    return "bad";
  }
  return "warn";
}

export default App;
