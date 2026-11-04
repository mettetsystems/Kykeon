import { validateCustomConfig } from "./custom-nodes";
import { optimizationNodeIds } from "./performance";
import {
  syntheticMetrics,
  comparisonReport,
  acceptanceReport,
} from "./performance";
import { definition, nodeDefinition, samplingFields } from "./catalog";
import type {
  FlowNode,
  Workflow,
  ModelReference,
  ExecutionEngine,
  RunResult,
  CustomModelManifest,
} from "./types";
export function createNode(definitionId: string, x = 100, y = 100): FlowNode {
  const d = definition(definitionId);
  return {
    id: crypto.randomUUID(),
    type: "workbench",
    position: { x, y },
    data: {
      definitionId,
      config: Object.fromEntries(d.fields.map((f) => [f.key, f.default])),
    },
  };
}
export type StarterKind =
  | "inference"
  | "merge"
  | "quantize"
  | "optimization"
  | "validation"
  | "rescale";
export function starter(kind: StarterKind): Workflow {
  const ids =
    kind === "rescale"
      ? ["model-2", "operation-4", "evaluate", "export"]
      : kind === "optimization" || kind === "validation"
      ? [
          "model-2",
          "runtime-config",
          ...(kind === "optimization"
            ? ["opt-kv-cache", "opt-batching", "opt-context"]
            : []),
          "workload",
          "benchmark",
          kind === "optimization" ? "optimization-report" : "validation-gate",
        ]
      : kind === "inference"
        ? ["model-2", "prompt", "sampling", "generate", "compare"]
        : kind === "merge"
          ? ["model-2", "model-2", "surgery-25", "evaluate", "export"]
          : ["model-2", "surgery-11", "evaluate", "export"];
  const nodes = ids.map((id, i) => createNode(id, 70 + i * 300, 100));
  if (kind === "merge") {
    nodes[0].position = { x: 50, y: 30 };
    nodes[1].position = { x: 50, y: 280 };
    nodes[1].data.config.name = "Kykeon Code 1B";
    nodes[1].data.config.path = "/models/kykeon-code-1b";
    nodes
      .slice(2)
      .forEach((n, i) => (n.position = { x: 380 + i * 300, y: 140 }));
  }
  const pairs =
    kind === "merge"
      ? [
          [0, 2, "a"],
          [1, 2, "b"],
          [2, 3, "input"],
          [3, 4, "input"],
        ]
      : nodes.slice(1).map((_, i) => [i, i + 1, "input"]);
  return {
    version: 1,
    name:
      kind === "optimization"
        ? "Optimize inference serving"
        : kind === "validation"
          ? "Validate performance budgets"
          : kind === "inference"
            ? "Explore text generation"
      : kind === "merge"
        ? "Blend two model checkpoints"
        : kind === "rescale"
          ? "Rescale one weight tensor"
          : "Make a smaller model",
    mode:
      kind === "optimization" || kind === "validation"
        ? kind
        : kind === "inference"
          ? "inference"
          : "surgery",
    nodes,
    edges: pairs.map(([a, b, h], i) => ({
      id: `edge-${i}`,
      source: nodes[Number(a)].id,
      target: nodes[Number(b)].id,
      sourceHandle: "out",
      targetHandle: String(h),
      type: "smoothstep",
    })),
  };
}
export function graphOrder(w: Workflow): FlowNode[] {
  const seen = new Set<string>(),
    ordered: FlowNode[] = [];
  while (ordered.length < w.nodes.length) {
    const ready = w.nodes.filter(
      (n) =>
        !seen.has(n.id) &&
        w.edges
          .filter((e) => e.target === n.id)
          .every((e) => seen.has(e.source)),
    );
    if (!ready.length)
      throw Error("Connections form a cycle. Remove the loop before running.");
    ready.forEach((n) => {
      seen.add(n.id);
      ordered.push(n);
    });
  }
  return ordered;
}
export function connectionError(
  w: Workflow,
  source: string,
  target: string,
  handle: string,
): string | null {
  const a = w.nodes.find((n) => n.id === source),
    b = w.nodes.find((n) => n.id === target);
  if (!a || !b) return "Missing node.";
  if (source === target) return "A node cannot connect to itself.";
  if (nodeDefinition(a.data).output !== nodeDefinition(b.data).inputs[handle])
    return "These port types are incompatible.";
  if (w.edges.some((e) => e.target === target && e.targetHandle === handle))
    return "This input already has a connection.";
  try {
    graphOrder({
      ...w,
      edges: [
        ...w.edges,
        { id: "check", source, target, targetHandle: handle },
      ],
    });
  } catch {
    return "This connection would create a cycle.";
  }
  return null;
}
export function validateWorkflow(w: Workflow): void {
  if (!w.nodes.length) throw Error("Add nodes before running.");
  for (const n of w.nodes) {
    const d = nodeDefinition(n.data);
    if (!d.supported)
      throw Error(
        d.id === "operation-4"
          ? `${d.name} has no simulation adapter. Switch to Local execution, or remove it to run a simulation.`
          : `${d.name} is catalog-only. Remove it to run this demo.`,
      );
    for (const h of Object.keys(d.inputs))
      if (!w.edges.some((e) => e.target === n.id && e.targetHandle === h))
        throw Error(`${d.name}: connect the ${h} input.`);
    for (const f of d.fields) {
      const value = n.data.config[f.key];
      if (
        typeof f.default === "number" &&
        (typeof value !== "number" ||
          !Number.isFinite(value) ||
          (f.min !== undefined && value < f.min) ||
          (f.max !== undefined && value > f.max))
      )
        throw Error(`${d.name}: ${f.label} is outside its allowed range.`);
      if (f.options && !f.options.includes(String(value)))
        throw Error(`${d.name}: invalid ${f.label}.`);
    }
  }
  for (const e of w.edges) {
    if (e.sourceHandle !== "out") throw Error("Invalid source port.");
    const err = connectionError(
      { ...w, edges: w.edges.filter((x) => x.id !== e.id) },
      e.source,
      e.target,
      e.targetHandle || "",
    );
    if (err) throw Error(err);
  }
  graphOrder(w);
}
export function parseWorkflow(text: string): Workflow {
  const w = JSON.parse(text);
  delete w.execution;
  if (w.version !== 1)
    throw Error("Unsupported workflow version. Expected version 1.");
  if (
    typeof w.name !== "string" ||
    !["inference", "surgery", "optimization", "validation"].includes(w.mode) ||
    !Array.isArray(w.nodes) ||
    !Array.isArray(w.edges) ||
    w.nodes.length > 500 ||
    w.edges.length > 2000
  )
    throw Error("Invalid workflow format.");
  const ids = new Set<string>();
  for (const n of w.nodes) {
    if (
      typeof n.id !== "string" ||
      ids.has(n.id) ||
      n.type !== "workbench" ||
      !Number.isFinite(n.position?.x) ||
      !Number.isFinite(n.position?.y) ||
      !n.data?.config ||
      Array.isArray(n.data.config) ||
      typeof n.data.config !== "object"
    )
      throw Error("Invalid or duplicate node.");
    ids.add(n.id);
    if (n.data.definitionId === "operation-4") {
      const operation = definition("operation-4");
      for (const field of operation.fields)
        if (!Object.hasOwn(n.data.config, field.key))
          n.data.config[field.key] = field.default;
    }
    if (n.data.definitionId === "custom-code")
      validateCustomConfig(n.data.config);
    const d = nodeDefinition(n.data);
    for (const f of d.fields) {
      // Add only newly projected fields when restoring pre-existing workflows.
      if (f.projected && !Object.hasOwn(n.data.config, f.key))
        n.data.config[f.key] = f.default;
      if (typeof n.data.config[f.key] !== typeof f.default)
        throw Error(`Invalid field: ${f.label}`);
      if (
        f.projected &&
        ((f.options && !f.options.includes(String(n.data.config[f.key]))) ||
          (typeof f.default === "number" &&
            (!Number.isFinite(n.data.config[f.key]) ||
              (f.min !== undefined && Number(n.data.config[f.key]) < f.min) ||
              (f.max !== undefined && Number(n.data.config[f.key]) > f.max))))
      )
        throw Error(`Invalid projected field: ${f.label}`);
    }
    if (n.data.definitionId === "custom" && !validModel(n.data.model))
      throw Error("Invalid custom model.");
  }
  const edgeIds = new Set<string>();
  for (const e of w.edges) {
    if (
      typeof e.id !== "string" ||
      edgeIds.has(e.id) ||
      !ids.has(e.source) ||
      !ids.has(e.target) ||
      e.sourceHandle !== "out"
    )
      throw Error("Invalid edge.");
    edgeIds.add(e.id);
    const err = connectionError(
      { ...w, edges: w.edges.filter((x: { id: string }) => x.id !== e.id) },
      e.source,
      e.target,
      e.targetHandle,
    );
    if (err) throw Error(err);
  }
  graphOrder(w);
  return w;
}
export function validModel(m: unknown): m is ModelReference {
  if (!m || typeof m !== "object") return false;
  const a = m as ModelReference;
  return (
    ["name", "path", "architecture"].every(
      (k) => typeof a[k as keyof ModelReference] === "string",
    ) &&
    Number.isFinite(a.sizeGB) &&
    a.sizeGB > 0 &&
    Array.isArray(a.history) &&
    a.history.every((x) => typeof x === "string") &&
    Array.isArray(a.sources) &&
    a.sources.every((x) => typeof x === "string")
  );
}
export function parseManifests(value: unknown): CustomModelManifest[] {
  if (
    !Array.isArray(value) ||
    !value.every(
      (m) =>
        m.version === 1 &&
        m.simulated === true &&
        typeof m.id === "string" &&
        validModel(m.model),
    )
  )
    throw Error("Saved custom models are invalid.");
  return value;
}
export function historyPush<T>(past: T[], current: T) {
  return [...past, current].slice(-50);
}
export function simulateText(
  prompt: string,
  c: Record<string, string | number>,
): string {
  const topic = /quant|smaller|precision/i.test(prompt)
    ? "Quantization represents model weights with fewer bits, reducing storage while potentially changing output quality."
    : /merge|blend/i.test(prompt)
      ? "Model merging combines compatible checkpoints using weighted contributions from each source."
      : /language|token|model/i.test(prompt)
        ? "A language model predicts the next token from the context, then repeats this process to build a response."
        : `This is an illustrative response to your prompt: ${prompt}.`;
  let hash = Number(c.seed) || 0;
  for (const ch of prompt)
    hash = (Math.imul(hash, 31) + ch.charCodeAt(0)) >>> 0;
  const options = [
    "Think of it as choosing the next step from a ranked set of possibilities.",
    "Each choice changes the context for the following step.",
    "Adjusting the controls changes which candidates can be selected.",
    "The workflow makes each stage visible and repeatable.",
  ];
  const diversity = Math.max(
    1,
    Math.min(4, Math.ceil((Number(c.topK) / 15) * Number(c.topP))),
  );
  const pick =
    Number(c.temperature) === 0
      ? 0
      : (hash + Math.round(Number(c.temperature) * 10)) % diversity;
  let text = topic + " " + options[pick];
  if (Number(c.repetition) <= 1) text += " " + options[pick];
  text = text
    .split(/\s+/)
    .slice(0, Number(c.maxTokens) || 80)
    .join(" ");
  for (const stop of String(c.stop || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)) {
    const i = text.indexOf(stop);
    if (i >= 0) text = text.slice(0, i);
  }
  return text;
}
const pause = (signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal.aborted) return reject(Error("Run cancelled."));
    const abort = () => {
      clearTimeout(timer);
      reject(Error("Run cancelled."));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, 350);
    signal.addEventListener("abort", abort, { once: true });
  });
export const simulationEngine: ExecutionEngine = {
  async run(w, onEvent, signal) {
    validateWorkflow(w);
    const values: Record<string, any> = {};
    let result: RunResult = { values };
    for (const n of graphOrder(w)) {
      if (signal.aborted) throw Error("Run cancelled.");
      const d = nodeDefinition(n.data),
        c = n.data.config;
      onEvent({ nodeId: n.id, state: "running", detail: d.name });
      await pause(signal);
      const inputs = Object.fromEntries(
        w.edges
          .filter((e) => e.target === n.id)
          .map((e) => [e.targetHandle!, values[e.source]]),
      );
      let v: any;
      if (d.id === "model-2")
        v = {
          name: String(c.name),
          path: String(c.path),
          architecture: String(c.architecture),
          sizeGB: 2,
          history: [],
          sources: [String(c.path)],
        } satisfies ModelReference;
      else if (d.id === "custom") {
        if (!validModel(n.data.model))
          throw Error("Custom model metadata is missing.");
        v = structuredClone(n.data.model);
      } else if (d.id === "prompt")
        v = {
          model: inputs.input,
          prompt: String(c.prompt),
          settings: Object.fromEntries(
            samplingFields.map((f) => [f.key, f.default]),
          ),
        };
      else if (d.output === "context")
        v = { ...inputs.input, settings: { ...inputs.input.settings, ...c } };
      else if (d.id === "generate") {
        v = {
          text: simulateText(inputs.input.prompt, inputs.input.settings),
          baseline: simulateText(
            inputs.input.prompt,
            Object.fromEntries(samplingFields.map((f) => [f.key, f.default])),
          ),
        };
        result.text = v.text;
      } else if (d.id === "compare") {
        v = inputs.input;
        result.report = `Default fixture\n${v.baseline}\n\nConfigured fixture\n${v.text}`;
      } else if (d.id === "runtime-config") {
        v = { model: inputs.input, settings: { ...c } };
      } else if (optimizationNodeIds.has(d.id)) {
        v = { ...inputs.input, settings: { ...inputs.input.settings, ...c } };
      } else if (d.id === "workload") {
        if (Number(c.inputTokens) > Number(inputs.input.settings.context))
          throw Error(
            "Workload input tokens exceed the runtime context budget.",
          );
        v = { ...inputs.input, load: { ...c } };
      } else if (d.id === "benchmark") {
        v = {
          configured: syntheticMetrics(inputs.input),
          baseline: syntheticMetrics(inputs.input, true),
        };
        result.report = comparisonReport(v);
      } else if (d.id === "optimization-report") {
        v = inputs.input;
        result.report = comparisonReport(v);
      } else if (d.id === "validation-gate") {
        v = inputs.input;
        result.report = acceptanceReport(v.configured, c);
      } else if (d.id === "surgery-25") {
        const { a, b } = inputs;
        if (a.architecture !== b.architecture)
          throw Error("Merge requires matching source architectures.");
        v = {
          ...a,
          name: `${a.name} blend`,
          path: "",
          sources: [...a.sources, ...b.sources],
          history: [
            ...a.history,
            ...b.history,
            `Linear merge: A ${c.weight}, B ${1 - Number(c.weight)}`,
          ],
          sizeGB:
            a.sizeGB * Number(c.weight) + b.sizeGB * (1 - Number(c.weight)),
        };
      } else if (d.id === "surgery-11") {
        const factor =
          c.precision === "INT4" ? 0.25 : c.precision === "INT8" ? 0.5 : 1;
        v = {
          ...inputs.input,
          name: `${inputs.input.name} ${c.precision}`,
          path: "",
          sizeGB: inputs.input.sizeGB * factor,
          history: [...inputs.input.history, `Quantize to ${c.precision}`],
        };
      } else {
        v = inputs.input;
        if (d.id === "evaluate")
          result.report = `Illustrative comparison · not measured\nBaseline fixture: 2.00 GB\nTransformed fixture: ${v.sizeGB.toFixed(2)} GB\nQuality: ${v.history.some((h: string) => h.includes("INT4")) ? "potential precision tradeoff" : "requires real evaluation"}\n${v.history.join("\n")}`;
        if (d.id === "export") result.model = v;
      }
      values[n.id] = v;
      onEvent({
        nodeId: n.id,
        state: "complete",
        detail: `${d.name} · simulated`,
      });
    }
    return result;
  },
};
