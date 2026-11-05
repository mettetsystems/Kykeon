import { capabilityLabels } from "./capabilities";
import { LocalCheckpointPanel, TensorDetails, formatBytes } from "./local-panel";
import {
  createLocalEngine,
  inflightJob,
  resumeLocalJob,
  readJson,
  type LocalDevice,
  type LocalRegistration,
} from "./local-engine";
import { readStorage, writeStorage } from "./storage";
import { CustomNodeDialog } from "./custom-nodes";
import {
  surgeryOperations,
  tierLabels,
  localScope,
} from "./surgery-operations";
import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  createContext,
  useContext,
} from "react";
import { createRoot } from "react-dom/client";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  applyNodeChanges,
  applyEdgeChanges,
  type NodeProps,
  type Connection,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  Activity,
  ArrowDownToLine,
  ArrowUpFromLine,
  Box,
  ChevronRight,
  CircleHelp,
  Copy,
  FlaskConical,
  GitBranch,
  Layers,
  Play,
  Plus,
  Redo2,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  Trash2,
  Undo2,
  X,
  Check,
  PanelLeftClose,
} from "lucide-react";
import { catalog, nodeDefinition, utilities } from "./catalog";
import { performanceNodes } from "./performance";
import type { StarterKind } from "./engine";
import { nodeHelp } from "./node-help";
import {
  createNode,
  starter,
  connectionError,
  parseWorkflow,
  parseManifests,
  simulationEngine,
  historyPush,
} from "./engine";
import type {
  Workflow,
  FlowNode,
  RunResult,
  CustomModelManifest,
  Field,
} from "./types";
import "@xyflow/react/dist/style.css";
import "./style.css";
const STORE = "kykeon-workspace-v1";
function load() {
  try {
    const raw = readStorage(STORE);
    if (!raw) return null;
    const s = JSON.parse(raw);
    return {
      workflow: parseWorkflow(JSON.stringify(s.workflow)),
      custom: parseManifests(s.custom),
      saved: (s.saved || []).map((w: unknown) =>
        parseWorkflow(JSON.stringify(w)),
      ),
    };
  } catch (e) {
    return {
      error: `Could not restore saved workspace: ${(e as Error).message}`,
    };
  }
}
const initial = load();
const NodeContext = createContext<{
  update: (id: string, key: string, value: string | number) => void;
  statuses: Record<string, string>;
  executionMode: "simulation" | "local";
}>({ update: () => {}, statuses: {}, executionMode: "simulation" });
function FieldInput({
  field,
  value,
  onChange,
  compact = false,
}: {
  field: Field;
  value: string | number;
  onChange: (v: string | number) => void;
  compact?: boolean;
}) {
  return (
    <label className={`field ${compact ? "compact" : ""}`}>
      <span>
        {field.label}
        {field.projected && <em className="planned-field-label">Planned</em>}
      </span>
      {field.options ? (
        <select
          className="nodrag"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {field.options.map((o) => (
            <option key={o}>{o}</option>
          ))}
        </select>
      ) : field.key === "prompt" || field.multiline ? (
        <textarea
          className="nodrag nowheel"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <input
          className="nodrag"
          type={typeof field.default === "number" ? "number" : "text"}
          min={field.min}
          max={field.max}
          step={field.step}
          value={value}
          onChange={(e) =>
            onChange(
              typeof field.default === "number"
                ? e.target.valueAsNumber
                : e.target.value,
            )
          }
        />
      )}
      {!compact && <small>{field.help}</small>}
    </label>
  );
}
function WorkbenchNode({ id, data, selected }: NodeProps<FlowNode>) {
  const d = nodeDefinition(data),
    ctx = useContext(NodeContext),
    status = ctx.statuses[id];
  return (
    <div
      className={`flow-node ${d.kind} ${selected ? "selected" : ""} ${status || ""}`}
    >
      <div className="node-heading">
        <span className="node-icon">
          {d.kind === "model" ? (
            <Box size={16} />
          ) : d.kind === "surgery" ? (
            <FlaskConical size={16} />
          ) : (
            <SlidersHorizontal size={16} />
          )}
        </span>
        <strong>{data.model?.name || d.name}</strong>
        <span className="node-light" />
      </div>
      <div className="node-body">
        <div className="node-badge">
          {d.id === "operation-4"
            ? "LOCAL ONLY"
            : d.supported && capabilityLabels(d).includes("Real local execution available")
              ? "SIMULATION · LOCAL"
              : d.supported
                ? "SIMULATED"
                : "CATALOG ONLY"}
        </div>
        {d.resourceProfile && (
          <span className={`resource-badge ${d.resourceProfile.tier}`}>
            {tierLabels[d.resourceProfile.tier]}
          </span>
        )}
        {d.fields.slice(0, d.id === "sampling" ? 3 : 2).map((f) => (
          <FieldInput
            key={f.key}
            field={f}
            value={data.config[f.key]}
            compact
            onChange={(v) => ctx.update(id, f.key, v)}
          />
        ))}
        {!d.fields.length && <p>{d.description}</p>}
        {d.fields.length > 2 && (
          <small className="more-settings">
            Select to see all {d.fields.length} settings
          </small>
        )}
      </div>
      <div className="port-row">
        <div>
          {Object.entries(d.inputs).map(([h, t], i) => (
            <div className="port-input" key={h}>
              <Handle
                id={h}
                type="target"
                position={Position.Left}
                style={{
                  top:
                    Object.keys(d.inputs).length > 1
                      ? `${45 + i * 30}%`
                      : "70%",
                }}
              />
              <span>{h === "input" ? t : `${h.toUpperCase()} · ${t}`}</span>
            </div>
          ))}
        </div>
        <span>
          {d.output}
          <Handle
            id="out"
            type="source"
            position={Position.Right}
            style={{ top: "70%" }}
          />
        </span>
      </div>
      {status && (
        <div className="node-status">
          {status === "complete"
            ? "✓ Complete"
            : ctx.executionMode === "local"
              ? "◌ Running locally…"
              : "◌ Simulating…"}
        </div>
      )}
    </div>
  );
}
const nodeTypes = { workbench: WorkbenchNode };
function upstreamPath(workflow: Workflow, nodeId: string): string {
  const seen = new Set<string>();
  const walk = (id: string): string => {
    if (seen.has(id)) return "";
    seen.add(id);
    const node = workflow.nodes.find((item) => item.id === id);
    if (!node) return "";
    if (node.data.definitionId === "model-2")
      return String(node.data.config.path || "");
    for (const edge of workflow.edges.filter((item) => item.target === id)) {
      const found = walk(edge.source);
      if (found) return found;
    }
    return "";
  };
  return walk(nodeId);
}
function download(name: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function App() {
  const [customEditor, setCustomEditor] = useState<null | { id?: string }>(
    null,
  );
  const [workflow, setWorkflow] = useState<Workflow>(
    () => initial?.workflow ?? starter("inference"),
  );
  const [custom, setCustom] = useState<CustomModelManifest[]>(
    () => initial?.custom ?? [],
  );
  const [saved, setSaved] = useState<Workflow[]>(() => initial?.saved ?? []);
  const [notice, setNotice] = useState(
    initial && "error" in initial ? initial.error : "",
  );
  const [past, setPast] = useState<Workflow[]>([]),
    [future, setFuture] = useState<Workflow[]>([]);
  const [selected, setSelected] = useState<string | null>(null),
    [query, setQuery] = useState(""),
    [filter, setFilter] = useState("model"),
    [category, setCategory] = useState("all");
  const [statuses, setStatuses] = useState<Record<string, string>>({}),
    [result, setResult] = useState<RunResult | null>(null),
    [running, setRunning] = useState(false),
    [showLibrary, setShowLibrary] = useState(true),
    [showResults, setShowResults] = useState(true);
  const [runPhase, setRunPhase] = useState("");
  const [executionMode, setExecutionMode] = useState<"simulation" | "local">(
    "simulation",
  );
  const [deviceId, setDeviceId] = useState("");
  const [devices, setDevices] = useState<LocalDevice[]>([]);
  const [registrations, setRegistrations] = useState<LocalRegistration[]>([]);
  const [backend, setBackend] = useState("unknown");
  const [checkpointPath, setCheckpointPath] = useState("");
  const [preflight, setPreflight] = useState<Record<string, unknown> | null>(
    null,
  );
  const flow = useRef<ReactFlowInstance<FlowNode> | null>(null),
    abort = useRef<AbortController | null>(null),
    file = useRef<HTMLInputElement>(null),
    dragSnapshot = useRef<Workflow | null>(null);
  const commit = useCallback(
    (next: Workflow) => {
      if (running) return;
      setPast((p) => historyPush(p, workflow));
      setFuture([]);
      setWorkflow(next);
      setStatuses({});
      setResult(null);
    },
    [workflow, running],
  );
  const update = useCallback(
    (id: string, key: string, value: string | number) =>
      commit({
        ...workflow,
        nodes: workflow.nodes.map((n) =>
          n.id === id
            ? {
                ...n,
                data: { ...n.data, config: { ...n.data.config, [key]: value } },
              }
            : n,
        ),
      }),
    [workflow, commit],
  );
  useEffect(() => {
    const device = devices.find((item) => item.id === deviceId);
    if (!device?.uuid) return;
    const failure = writeStorage(
      "kykeon-device-preference",
      JSON.stringify({ uuid: device.uuid, name: device.name, id: device.id }),
    );
    if (failure)
      setNotice(`${failure} Download the workflow if you need a copy.`);
  }, [deviceId, devices]);
  async function refreshBackend() {
    try {
      await readJson(await fetch("/api/v1/health"));
      const deviceBody = await readJson<{ devices: LocalDevice[] }>(
        await fetch("/api/v1/devices"),
      );
      const modelBody = await readJson<{ models: LocalRegistration[] }>(
        await fetch("/api/v1/models"),
      );
      setDevices(deviceBody.devices);
      setRegistrations(modelBody.models);
      const saved = readStorage("kykeon-device-preference");
      if (saved) {
        try {
          const preference = JSON.parse(saved) as { uuid?: string };
          const match = deviceBody.devices.find(
            (device) => preference.uuid && device.uuid === preference.uuid,
          );
          setDeviceId(match?.id || "");
        } catch {
          setDeviceId("");
        }
      }
      setBackend("ok");
    } catch {
      setBackend("unavailable");
    }
  }
  useEffect(() => {
    refreshBackend();
  }, []);
  useEffect(() => {
    const node = workflow.nodes.find((item) => item.id === selected);
    if (!node || node.data.definitionId !== "operation-4" || backend !== "ok") {
      setPreflight(null);
      return;
    }
    const path = upstreamPath(workflow, node.id);
    const registration = registrations.find((item) => item.resolved_path === path);
    const tensor = String(node.data.config.tensor || "");
    if (!registration || !tensor) {
      setPreflight(null);
      return;
    }
    const timer = window.setTimeout(() => {
      fetch(`/api/v1/models/${registration.id}/preflight`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Kykeon-Request": "1",
        },
        body: JSON.stringify({
          prompt: String(node.data.config.prompt || ""),
          max_new_tokens: Number(node.data.config.maxTokens || 80),
          tensor,
        }),
      })
        .then((response) => (response.ok ? response.json() : null))
        .then((body) => setPreflight(body))
        .catch(() => setPreflight(null));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [selected, workflow, backend, registrations]);
  useEffect(() => {
    const failure = writeStorage(STORE, JSON.stringify({ workflow, custom, saved }));
    if (failure) setNotice(`${failure} Download your workflow to keep a copy.`);
  }, [workflow, custom, saved]);
  const undo = useCallback(() => {
    if (!past.length || running) return;
    setFuture((f) => [workflow, ...f]);
    setWorkflow(past[past.length - 1]);
    setPast((p) => p.slice(0, -1));
    setResult(null);
    setStatuses({});
  }, [past, workflow, running]);
  const redo = useCallback(() => {
    if (!future.length || running) return;
    setPast((p) => historyPush(p, workflow));
    setWorkflow(future[0]);
    setFuture((f) => f.slice(1));
    setResult(null);
    setStatuses({});
  }, [future, workflow, running]);
  function remove() {
    if (!selected) return;
    commit({
      ...workflow,
      nodes: workflow.nodes.filter((n) => n.id !== selected),
      edges: workflow.edges.filter(
        (e) => e.source !== selected && e.target !== selected,
      ),
    });
    setSelected(null);
  }
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (customEditor) return;
      if ((e.target as HTMLElement).closest("input,textarea,select")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        remove();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });
  function add(id: string, model?: CustomModelManifest) {
    const pos = flow.current?.screenToFlowPosition({
      x: window.innerWidth * 0.5,
      y: window.innerHeight * 0.43,
    }) || { x: 150, y: 150 };
    const n = createNode(id, pos.x, pos.y);
    if (model) n.data.model = structuredClone(model.model);
    commit({ ...workflow, nodes: [...workflow.nodes, n] });
    setSelected(n.id);
  }
  function chooseStarter(kind: StarterKind) {
    commit(starter(kind));
    setFilter(
      kind === "optimization" || kind === "validation" ? kind : "model",
    );
    setCategory("all");
    setQuery("");
    setSelected(null);
    setNotice(
      kind === "rescale"
        ? "Weight rescaling runs only in Local execution. Loading this starter does not switch modes."
        : "",
    );
    setExecutionMode("simulation");
    setTimeout(
      () =>
        flow.current?.setViewport(
          { x: 30, y: 20, zoom: 0.8 },
          { duration: 300 },
        ),
      60,
    );
  }
  function connect(c: Connection) {
    const err = connectionError(
      workflow,
      c.source,
      c.target,
      c.targetHandle || "",
    );
    if (err) {
      setNotice(err);
      return;
    }
    commit({
      ...workflow,
      edges: [
        ...workflow.edges,
        { ...c, id: crypto.randomUUID(), type: "smoothstep" },
      ],
    });
  }
  async function run() {
    setRunning(true);
    setStatuses({});
    setResult(null);
    setNotice("");
    setShowResults(true);
    const controller = new AbortController();
    abort.current = controller;
    try {
      const engine =
        executionMode === "local"
          ? createLocalEngine(deviceId, { onPhase: setRunPhase })
          : simulationEngine;
      const onEvent = (e: { nodeId: string; state: string }) => setStatuses((s) => ({ ...s, [e.nodeId]: e.state }));
      const r = inflightJob()
        ? await resumeLocalJob(onEvent, controller.signal, { onPhase: setRunPhase })
        : await engine.run(workflow, onEvent, controller.signal);
      setResult(r);
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setRunning(false);
      setRunPhase("");
      setStatuses((s) =>
        Object.fromEntries(
          Object.entries(s).filter(([, v]) => v === "complete"),
        ),
      );
    }
  }
  useEffect(() => {
    // The timer avoids duplicate monitoring during StrictMode's effect probe.
    // Monitoring an existing job never enables Local execution or submits work.
    const timer = window.setTimeout(() => { if (inflightJob()) void run(); }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  function publish() {
    if (!result?.model) return;
    if (result.values.execution === "local") {
      const registration = registrations.find(
        (item) => item.resolved_path === result.model?.path,
      );
      setNotice(
        registration
          ? "Derived checkpoint is registered. Placing it on the canvas does not claim the model improved."
          : "Derived checkpoint written and reloaded. Register the artifact path to use it again. This is not a quality score.",
      );
      refreshBackend();
      return;
    }
    const m: CustomModelManifest = {
      version: 1,
      simulated: true,
      id: crypto.randomUUID(),
      model: structuredClone(result.model),
    };
    setCustom((c) => [...c, m]);
    download("kykeon-custom-model.json", m);
    setNotice(
      "Simulated model added to Your models. Manifest downloaded; no checkpoint was written.",
    );
  }
  const active = workflow.nodes.find((n) => n.id === selected),
    def = active ? nodeDefinition(active.data) : null;
  const help = def ? nodeHelp(def) : null;
  const pool =
    filter === "utility"
      ? utilities.filter((d) => d.id !== "custom")
      : [...catalog, ...performanceNodes, ...surgeryOperations].filter(
          (d) => d.kind === filter,
        );
  const shown = pool.filter(
    (d) =>
      (category === "all" || d.category === category) &&
      `${d.name} ${d.description} ${d.resourceProfile ? tierLabels[d.resourceProfile.tier] : ""}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <NodeContext.Provider value={{ update, statuses, executionMode }}>
      <div className="app">
        <header>
          <a className="brand" href="#" aria-label="Kykeon home">
            <span className="brand-mark">ϰ</span>
            <b>
              kykeon<span>MODEL WORKBENCH</span>
            </b>
          </a>
          <nav>
            <button
              className={workflow.mode === "inference" ? "active" : ""}
              disabled={running}
              onClick={() => chooseStarter("inference")}
            >
              <GitBranch size={16} />
              Inference
            </button>
            <button
              className={workflow.mode === "surgery" ? "active" : ""}
              disabled={running}
              onClick={() => chooseStarter("merge")}
            >
              <FlaskConical size={16} />
              Model Surgery
            </button>
            <button
              className={workflow.mode === "optimization" ? "active" : ""}
              disabled={running}
              onClick={() => chooseStarter("optimization")}
            >
              <SlidersHorizontal size={16} />
              Inference Optimization
            </button>
            <button
              className={workflow.mode === "validation" ? "active" : ""}
              disabled={running}
              onClick={() => chooseStarter("validation")}
            >
              <Activity size={16} />
              Performance Validation
            </button>
          </nav>
          <div className="header-end">
            <span className="prototype">
              <i />
              {executionMode === "local" ? "Local execution" : "Simulation"}
            </span>
            <button
              className="icon"
              title="About execution modes"
              onClick={() =>
                setNotice(
                  executionMode === "local"
                    ? "Local execution loads a registered checkpoint and does not fall back to simulation. Simulation remains available from the mode switch."
                    : "Simulation uses educational fixtures. Local paths are not opened. Switch to Local execution to run a registered checkpoint.",
                )
              }
            >
              <CircleHelp size={18} />
            </button>
          </div>
        </header>
        <div className="toolbar">
          <div className="workflow-title">
            <button
              className="icon"
              title="Toggle node library"
              onClick={() => setShowLibrary(!showLibrary)}
            >
              <PanelLeftClose size={17} />
            </button>
            <span className="crumb">Workspace</span>
            <ChevronRight size={14} />
            <input
              aria-label="Workflow name"
              value={workflow.name}
              disabled={running}
              onChange={(e) => commit({ ...workflow, name: e.target.value })}
            />
            <span className="version">v0.1</span>
          </div>
          <div className="toolbar-actions">
            <button
              className="icon"
              title="Undo (Ctrl+Z)"
              disabled={!past.length || running}
              onClick={undo}
            >
              <Undo2 size={17} />
            </button>
            <button
              className="icon"
              title="Redo (Ctrl+Shift+Z)"
              disabled={!future.length || running}
              onClick={redo}
            >
              <Redo2 size={17} />
            </button>
            <span className="separator" />
            <button
              disabled={running}
              onClick={() => {
                setSaved((s) => [
                  ...s.filter((w) => w.name !== workflow.name),
                  structuredClone(workflow),
                ]);
                setNotice("Workflow saved in this browser.");
              }}
            >
              <Save size={15} />
              <span>Save</span>
            </button>
            <button
              disabled={running}
              onClick={() => file.current?.click()}
              title="Import workflow"
            >
              <ArrowUpFromLine size={16} />
            </button>
            <button
              onClick={() => download("kykeon-workflow.json", workflow)}
              title="Export workflow"
            >
              <ArrowDownToLine size={16} />
            </button>
            <div className="mode-toggle" role="group" aria-label="Execution mode">
              <button
                className={executionMode === "simulation" ? "active" : ""}
                disabled={running}
                onClick={() => setExecutionMode("simulation")}
              >
                Simulation
              </button>
              <button
                className={executionMode === "local" ? "active" : ""}
                disabled={running}
                onClick={() => setExecutionMode("local")}
              >
                Local
              </button>
            </div>
            <button
              className="run-button"
              onClick={running ? () => abort.current?.abort() : run}
            >
              {running ? (
                <Square size={14} />
              ) : (
                <Play size={14} fill="currentColor" />
              )}
              {running
                ? runPhase === "cancel_pending"
                  ? "Stopping…"
                  : "Cancel"
                : executionMode === "local"
                  ? "Run locally"
                  : "Run simulation"}
            </button>
          </div>
        </div>
        {notice && (
          <div className="notice" role="status">
            <span>{notice}</span>
            <button
              className="icon"
              aria-label="Dismiss message"
              onClick={() => setNotice("")}
            >
              <X size={15} />
            </button>
          </div>
        )}
        <div className={`workspace ${showLibrary ? "" : "library-hidden"}`}>
          {showLibrary && (
            <aside className="library">
              <div className="panel-heading">
                <h2>Node library</h2>
                <span title="150 original entries plus 50 surgery operations">
                  200
                </span>
              </div>
              <div className="search">
                <Search size={15} />
                <input
                  aria-label="Search nodes"
                  placeholder="Find a node…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <kbd>⌕</kbd>
              </div>
              <div className="library-tabs">
                {[
                  ["model", "Models"],
                  ["control", "Controls"],
                  ["surgery", "Tools"],
                  ["operation", "Operations"],
                  ["utility", "Core"],
                  ["optimization", "Optimize"],
                  ["validation", "Validate"],
                ].map(([k, v]) => (
                  <button
                    key={k}
                    className={filter === k ? "active" : ""}
                    onClick={() => {
                      setFilter(k);
                      setCategory("all");
                    }}
                  >
                    {v}
                  </button>
                ))}
              </div>
              <select
                aria-label="Category"
                className="category-select"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              >
                <option value="all">All categories · {pool.length}</option>
                {[...new Set(pool.map((d) => d.category))].map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
              {filter === "operation" && (
                <p className="resource-legend">
                  5090 · ~3B · PEFT budget
                  <br />
                  Provisional scope flags; not benchmarked.
                </p>
              )}
              <LocalCheckpointPanel
                backend={backend}
                devices={devices}
                deviceId={deviceId}
                onDevice={setDeviceId}
                path={checkpointPath}
                onPath={setCheckpointPath}
                registrations={registrations}
                busy={running}
                onRefresh={refreshBackend}
                onRegister={async () => {
                  try {
                    await readJson(
                      await fetch("/api/v1/models/register", {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                          "X-Kykeon-Request": "1",
                        },
                        body: JSON.stringify({ path: checkpointPath }),
                      }),
                    );
                    await refreshBackend();
                    setNotice(
                      "Checkpoint registered from local files. No download was performed.",
                    );
                  } catch (error) {
                    setNotice((error as Error).message);
                  }
                }}
                onPlace={(registration) => {
                  const existing = workflow.nodes.find(
                    (node) => node.data.definitionId === "model-2",
                  );
                  if (existing) {
                    commit({
                      ...workflow,
                      nodes: workflow.nodes.map((node) =>
                        node.id === existing.id
                          ? {
                              ...node,
                              data: {
                                ...node.data,
                                config: {
                                  ...node.data.config,
                                  name: registration.name,
                                  path: registration.resolved_path,
                                  architecture: registration.architecture,
                                },
                              },
                            }
                          : node,
                      ),
                    });
                    setSelected(existing.id);
                    setNotice(
                      "The model node now points at the registered checkpoint.",
                    );
                    return;
                  }
                  const position = flow.current?.screenToFlowPosition({
                    x: window.innerWidth * 0.45,
                    y: window.innerHeight * 0.4,
                  }) || { x: 80, y: 80 };
                  const placed = createNode("model-2", position.x, position.y);
                  placed.data.config = {
                    name: registration.name,
                    path: registration.resolved_path,
                    architecture: registration.architecture,
                  };
                  commit({ ...workflow, nodes: [...workflow.nodes, placed] });
                  setSelected(placed.id);
                }}
              />
              <div className="catalog-scroll">
                <div className="section-label">
                  {query
                    ? "SEARCH RESULTS"
                    : filter === "utility"
                      ? "WORKFLOW ESSENTIALS"
                      : `EXPLORE ${filter.toUpperCase()}S`}
                  <span>{shown.length}</span>
                </div>
                {shown.map((d) => (
                  <button
                    className="catalog-item"
                    key={d.id}
                    disabled={running}
                    onClick={() => add(d.id)}
                    title={d.description}
                  >
                    <span className={`catalog-symbol ${d.kind}`}>
                      {d.kind === "model" ? (
                        <Box size={16} />
                      ) : d.kind === "surgery" ? (
                        <FlaskConical size={16} />
                      ) : (
                        <SlidersHorizontal size={16} />
                      )}
                    </span>
                    <span>
                      <strong>{d.name}</strong>
                      <small>
                        {d.id === "operation-4"
                          ? "Real local execution · no simulation"
                          : d.supported
                            ? "Simulation available"
                            : "Catalog only"}
                      </small>
                      {d.resourceProfile && (
                        <span
                          className={`resource-badge ${d.resourceProfile.tier}`}
                        >
                          {tierLabels[d.resourceProfile.tier]}
                        </span>
                      )}
                    </span>
                    <Plus size={14} />
                  </button>
                ))}
                {!shown.length && (
                  <p className="empty-small">No matching nodes.</p>
                )}
                {custom.length > 0 && (
                  <>
                    <div className="section-label">
                      YOUR MODELS<span>{custom.length}</span>
                    </div>
                    {custom.map((m) => (
                      <button
                        className="catalog-item"
                        key={m.id}
                        disabled={running}
                        onClick={() => add("custom", m)}
                      >
                        <Box size={16} />
                        <span>
                          <strong>{m.model.name}</strong>
                          <small>Custom · simulated</small>
                        </span>
                        <Plus size={14} />
                      </button>
                    ))}
                  </>
                )}
              </div>
              <div className="library-foot">
                <Layers size={16} />
                <span>
                  Explore freely.
                  <br />
                  <small>Your source models stay untouched.</small>
                </span>
              </div>
            </aside>
          )}
          <main className="canvas-column">
            <div className="canvas-top">
              <button
                className="add-custom-button"
                disabled={running}
                onClick={() => setCustomEditor({})}
              >
                <Plus size={14} />
                Add custom node
              </button>
              <span>
                <i className="green-dot" />{" "}
                {
                  {
                    inference: "INFERENCE WORKFLOW",
                    surgery: "MODEL SURGERY",
                    optimization: "INFERENCE OPTIMIZATION",
                    validation: "PERFORMANCE VALIDATION PIPELINE",
                  }[workflow.mode]
                }
              </span>
              <select
                aria-label="Load starter workflow"
                value=""
                disabled={running}
                onChange={(e) => chooseStarter(e.target.value as StarterKind)}
              >
                <option value="" disabled>
                  Starter workflows
                </option>
                <option value="inference">Explore text generation</option>
                <option value="merge">Blend two checkpoints</option>
                <option value="quantize">Make a smaller model</option>
                <option value="rescale">Rescale one weight tensor (local)</option>
                <option value="optimization">Optimize inference serving</option>
                <option value="validation">Validate performance budgets</option>
              </select>
              {saved.length > 0 && (
                <select
                  aria-label="Load saved workflow"
                  value=""
                  disabled={running}
                  onChange={(e) => {
                    commit(structuredClone(saved[Number(e.target.value)]));
                    setExecutionMode("simulation");
                    setTimeout(() => flow.current?.fitView(), 60);
                  }}
                >
                  <option value="" disabled>
                    Saved workflows
                  </option>
                  {saved.map((w, i) => (
                    <option key={i} value={i}>
                      {w.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="canvas">
              <ReactFlow
                nodes={workflow.nodes}
                edges={workflow.edges}
                nodeTypes={nodeTypes}
                onInit={(i) => {
                  flow.current = i;
                }}
                defaultViewport={{ x: 30, y: 20, zoom: 0.8 }}
                minZoom={0.2}
                maxZoom={1.6}
                nodesDraggable={!running}
                nodesConnectable={!running}
                edgesReconnectable={false}
                deleteKeyCode={null}
                onNodeClick={(_, n) => setSelected(n.id)}
                onPaneClick={() => setSelected(null)}
                onConnect={connect}
                onNodesChange={(changes) => {
                  if (running) return;
                  const next = {
                    ...workflow,
                    nodes: applyNodeChanges(changes, workflow.nodes),
                  };
                  if (changes.some((c) => c.type === "remove")) commit(next);
                  else setWorkflow(next);
                }}
                onNodeDragStart={() => {
                  dragSnapshot.current = structuredClone(workflow);
                }}
                onNodeDragStop={() => {
                  if (dragSnapshot.current) {
                    setPast((p) => historyPush(p, dragSnapshot.current!));
                    setFuture([]);
                    dragSnapshot.current = null;
                  }
                }}
                onEdgesChange={(changes) => {
                  if (!running && changes.some((c) => c.type === "remove"))
                    commit({
                      ...workflow,
                      edges: applyEdgeChanges(changes, workflow.edges),
                    });
                }}
                onEdgeDoubleClick={(_, edge) =>
                  commit({
                    ...workflow,
                    edges: workflow.edges.filter((e) => e.id !== edge.id),
                  })
                }
              >
                <Background color="#333b42" gap={22} size={1} />
                <Controls showInteractive={false} />
                <MiniMap
                  pannable
                  zoomable
                  nodeColor="#34453d"
                  maskColor="rgba(15,18,21,.75)"
                />
              </ReactFlow>
              <div className="canvas-caption">
                Connect ideas. Understand what changes.
                <span>
                  Drag to arrange · Scroll to zoom · Double-click an edge to
                  remove
                </span>
              </div>
            </div>
            <section className={`results ${showResults ? "" : "collapsed"}`}>
              <button
                className="results-heading"
                onClick={() => setShowResults(!showResults)}
              >
                <span>
                  <Activity size={16} />
                  Run output{" "}
                  <span className="simulation-label">
                    {result
                      ? result.values.execution === "local"
                        ? "LOCAL EXECUTION"
                        : "SIMULATION"
                      : executionMode === "local"
                        ? "LOCAL EXECUTION"
                        : "SIMULATION"}
                  </span>
                </span>
                <span>
                  {running
                    ? "Running…"
                    : result
                      ? "Completed"
                      : "Ready when you are"}
                  <ChevronRight size={16} />
                </span>
              </button>
              {showResults && (
                <div className="results-content">
                  {result ? (
                    <>
                      <div className="result-text">
                        <div className="section-label">
                          {result.values.execution === "local"
                            ? "LOCAL RESULT"
                            : result.text
                              ? "GENERATED FIXTURE"
                              : "MODEL COMPARISON"}
                          <span>
                            {result.values.execution === "local"
                              ? "Measured on the selected device"
                              : "Not a live model result"}
                          </span>
                        </div>
                        <pre>
                          {result.report ||
                            result.text ||
                            "Simulation complete."}
                        </pre>
                      </div>
                      {result.model && (
                        <div className="export-card">
                          <Check size={19} />
                          <strong>{result.model.name}</strong>
                          <small>
                            {result.values.execution === "local"
                              ? `${formatBytes(result.model.sizeGB * 1_000_000_000)} on disk · reloaded checkpoint`
                              : `${result.model.sizeGB.toFixed(2)} GB · illustrative size`}
                          </small>
                          <button className="accent-button" onClick={publish}>
                            <ArrowDownToLine size={14} />
                            {result.values.execution === "local"
                              ? "Record derived checkpoint"
                              : "Export custom model"}
                          </button>
                          <small>
                            {result.values.execution === "local"
                              ? "Weights are already in the artifact root. This does not claim an improvement."
                              : "Downloads metadata, not model weights."}
                          </small>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="output-empty">
                      <div className="empty-icon">
                        <Play size={20} />
                      </div>
                      <div>
                        <strong>
                          {running
                            ? "Walking through your workflow…"
                            : "A little experimentation goes a long way."}
                        </strong>
                        <p>
                          {running
                            ? `${Object.values(statuses).filter((s) => s === "complete").length} of ${workflow.nodes.length} nodes complete`
                            : "Adjust a control, run the simulation, and compare the result."}
                        </p>
                      </div>
                      <span className="key-hint">LOCAL · SIMULATED</span>
                    </div>
                  )}
                </div>
              )}
            </section>
          </main>
          <aside className="inspector">
            <div className="panel-heading">
              <h2>{active ? "Node inspector" : "Workflow guide"}</h2>
              <button
                className="icon"
                aria-label="Close node inspector"
                onClick={() => setSelected(null)}
              >
                <X size={16} />
              </button>
            </div>
            {active && def ? (
              <div className="inspector-scroll" key={active.id}>
                <div className={`inspector-icon ${def.kind}`}>
                  <Box size={22} />
                </div>
                <h3>{active.data.model?.name || def.name}</h3>
                <span className="support-badge">
                  {def.supported ? "Simulation available" : "Catalog reference"}
                </span>
                <div className="capability-row">
                  {capabilityLabels(def).map((label) => (
                    <span key={label}>{label}</span>
                  ))}
                </div>
                {active.data.definitionId === "custom-code" && (
                  <div className="custom-source-panel">
                    <p>
                      {String(active.data.config.format)} · Source stored only
                    </p>
                    <button
                      disabled={running}
                      onClick={() => setCustomEditor({ id: active.id })}
                    >
                      Edit source
                    </button>
                    <pre>{String(active.data.config.source)}</pre>
                  </div>
                )}
                {def.resourceProfile && (
                  <section
                    className={`resource-panel ${def.resourceProfile.tier}`}
                  >
                    <span
                      className={`resource-badge ${def.resourceProfile.tier}`}
                    >
                      {tierLabels[def.resourceProfile.tier]}
                    </span>
                    <h4>Level {def.resourceProfile.level} · Resource scope</h4>
                    <p>{def.resourceProfile.reason}</p>
                    <small>
                      {localScope} This flag is a provisional product-scope
                      classification, not proof of hardware impossibility or an
                      execution entitlement.
                    </small>
                  </section>
                )}
                {help && (
                  <div className="node-documentation" key={active.id}>
                    <section>
                      <h4>What it does</h4>
                      <p>{help.explanation}</p>
                    </section>
                    <section>
                      <h4>When to use it</h4>
                      <p>{help.usage}</p>
                    </section>
                    <section>
                      <h4>Where it fits</h4>
                      <p>{help.workflow}</p>
                    </section>
                    <details>
                      <summary>Considerations & limitations</summary>
                      <p>{help.considerations}</p>
                    </details>
                  </div>
                )}
                {def.kind === "surgery" && (
                  <div className="help-card projected-notice">
                    <strong>Projected configuration</strong>
                    <p>
                      These planned fields are saved with your workflow. They
                      are not backend-validated and do not affect simulation.
                      Only unmarked demo controls are active.
                    </p>
                  </div>
                )}
                {def.fields.length > 0 && (
                  <h4 className="settings-title">Node settings</h4>
                )}
                {!def.supported && def.id !== "operation-4" && (
                  <div className="help-card">
                    This entry documents a future integration. It can be placed
                    on the canvas, but must be removed before a simulated run.
                  </div>
                )}
                {def.id === "operation-4" && (
                  <div className="help-card">
                    Simulation does not rescale weights. Local execution rescales
                    one selected projection after you register a dense Llama
                    checkpoint. Projected fields on other tools stay planned.
                  </div>
                )}
                {def.fields.map((f) =>
                  def.id === "operation-4" && f.key === "tensor" ? (
                    <label className="field" key={f.key}>
                      <span>{f.label}</span>
                      <select
                        className="nodrag"
                        value={String(active.data.config.tensor || "")}
                        onChange={(event) =>
                          update(active.id, "tensor", event.target.value)
                        }
                      >
                        <option value="">Select one projection</option>
                        {(
                          registrations.find(
                            (item) =>
                              item.resolved_path ===
                              upstreamPath(workflow, active.id),
                          )?.eligible_tensors || []
                        ).map((tensor) => (
                          <option key={tensor.name} value={tensor.name}>
                            {tensor.name}
                          </option>
                        ))}
                      </select>
                      <small>{f.help}</small>
                      <TensorDetails
                        tensor={registrations
                          .find(
                            (item) =>
                              item.resolved_path ===
                              upstreamPath(workflow, active.id),
                          )
                          ?.tensors.find(
                            (tensor) => tensor.name === active.data.config.tensor,
                          )}
                      />
                      {preflight && (
                        <small>
                          Preflight estimate, not a measurement: about{" "}
                          {formatBytes(Number(preflight.estimated_resident_bytes))} resident.
                          Uncertainty: {String(preflight.uncertainty || "high")}.{" "}
                          {Array.isArray(preflight.assumptions)
                            ? String(preflight.assumptions[0] || "")
                            : ""}
                        </small>
                      )}
                      {(() => {
                        const registration = registrations.find(
                          (item) =>
                            item.resolved_path ===
                            upstreamPath(workflow, active.id),
                        );
                        if (!registration) return null;
                        return (
                          <small>
                            Estimated weight memory{" "}
                            {formatBytes(registration.memory_estimate.weight_bytes)}.{" "}
                            {registration.memory_estimate.assumptions[0]} Uncertainty:{" "}
                            {registration.memory_estimate.uncertainty}. This is not an
                            out-of-memory guarantee.
                          </small>
                        );
                      })()}
                    </label>
                  ) : (
                    <FieldInput
                      key={f.key}
                      field={f}
                      value={active.data.config[f.key]}
                      onChange={(v) => update(active.id, f.key, v)}
                    />
                  ),
                )}
                {active.data.model && (
                  <div className="help-card">
                    {active.data.model.history.map((h, i) => (
                      <p key={i}>{h}</p>
                    ))}
                  </div>
                )}
                <div className="node-actions">
                  <button
                    disabled={running}
                    onClick={() => {
                      const n = structuredClone(active);
                      n.id = crypto.randomUUID();
                      n.position = {
                        x: n.position.x + 40,
                        y: n.position.y + 60,
                      };
                      commit({ ...workflow, nodes: [...workflow.nodes, n] });
                      setSelected(n.id);
                    }}
                  >
                    <Copy size={14} />
                    Duplicate
                  </button>
                  <button disabled={running} onClick={remove}>
                    <Trash2 size={14} />
                    Delete
                  </button>
                </div>
              </div>
            ) : (
              <div className="inspector-scroll">
                <div className="guide-eyebrow">LEARN BY CONNECTING</div>
                <h3>
                  Your model.
                  <br />
                  Every lever, visible.
                </h3>
                <p className="description">
                  {workflow.mode === "optimization"
                    ? "Compare runtime configurations using the same synthetic workload. Inspect illustrative latency, throughput, and memory tradeoffs."
                    : workflow.mode === "validation"
                      ? "Build a repeatable performance pipeline. Define a workload, calculate synthetic metrics, and check explicit acceptance thresholds."
                      : "Trace the journey from a model to its output. Every node makes one step explicit."}
                </p>
                {[
                  [
                    "01",
                    "Choose a starting point",
                    "Load a sample model or explore the full catalog.",
                  ],
                  [
                    "02",
                    "Shape the workflow",
                    "Select a node to understand its controls and edit settings.",
                  ],
                  [
                    "03",
                    "Run and compare",
                    "Follow the simulation, then see what your changes affect.",
                  ],
                ].map(([num, title, body]) => (
                  <div className="guide-step" key={num}>
                    <span>{num}</span>
                    <div>
                      <strong>{title}</strong>
                      <p>{body}</p>
                    </div>
                  </div>
                ))}
                <div className="help-card">
                  <FlaskConical size={18} />
                  <strong>A safe place to experiment</strong>
                  <p>
                    {executionMode === "local"
                      ? "Local execution reads only registered checkpoints and writes derived weights under the artifact root. A saved checkpoint is not an improved model."
                      : "Simulation uses educational fixtures. It does not load or modify model weights."}
                  </p>
                </div>
                <button
                  className="starter-link"
                  disabled={running}
                  onClick={() =>
                    chooseStarter(
                      workflow.mode === "inference" ? "quantize" : "inference",
                    )
                  }
                >
                  Try{" "}
                  {workflow.mode === "inference"
                    ? "model quantization"
                    : "text generation"}
                  <ChevronRight size={16} />
                </button>
              </div>
            )}
          </aside>
        </div>
        <footer>
          <span>
            <i className="green-dot" />
            {executionMode === "local"
              ? backend === "ok"
                ? "Local backend connected"
                : "Local backend unavailable"
              : "Simulation"}{" "}
            <span className="footer-divider">/</span>{" "}
            {deviceId || "No device selected"}
          </span>
          <span>
            {workflow.nodes.length} nodes · {workflow.edges.length} connections
            <span className="footer-divider">/</span>Kykeon 0.1
          </span>
        </footer>
      </div>
      {customEditor && (
        <CustomNodeDialog
          mode={workflow.mode}
          initial={
            customEditor.id
              ? workflow.nodes.find((n) => n.id === customEditor.id)?.data
                  .config
              : undefined
          }
          onClose={() => setCustomEditor(null)}
          onSave={(config) => {
            if (customEditor.id) {
              const old = workflow.nodes.find((n) => n.id === customEditor.id)!;
              const portsChanged =
                old.data.config.inputType !== config.inputType ||
                old.data.config.outputType !== config.outputType;
              commit({
                ...workflow,
                nodes: workflow.nodes.map((n) =>
                  n.id === old.id ? { ...n, data: { ...n.data, config } } : n,
                ),
                edges: portsChanged
                  ? workflow.edges.filter(
                      (e) => e.source !== old.id && e.target !== old.id,
                    )
                  : workflow.edges,
              });
              if (portsChanged)
                setNotice("Port types changed. Reconnect this node’s edges.");
            } else {
              const position = flow.current?.screenToFlowPosition({
                x: window.innerWidth * 0.5,
                y: window.innerHeight * 0.43,
              }) || { x: 150, y: 100 };
              const node = createNode("custom-code", position.x, position.y);
              node.data.config = config;
              commit({ ...workflow, nodes: [...workflow.nodes, node] });
              setSelected(node.id);
            }
            setCustomEditor(null);
          }}
        />
      )}
      <input
        ref={file}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          try {
            const w = parseWorkflow(await f.text());
            commit(w);
            setExecutionMode("simulation");
            setSelected(null);
            setNotice("Workflow imported.");
            setTimeout(() => flow.current?.fitView(), 60);
          } catch (err) {
            setNotice(`Import failed: ${(err as Error).message}`);
          }
          e.target.value = "";
        }}
      />
    </NodeContext.Provider>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ReactFlowProvider>
      <App />
    </ReactFlowProvider>
  </React.StrictMode>,
);
