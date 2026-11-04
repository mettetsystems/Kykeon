import type { Definition, Field } from "./types";
const number = (
  key: string,
  label: string,
  value: number,
  min: number,
  max: number,
  help: string,
): Field => ({ key, label, default: value, min, max, step: 1, help });
export const defaultOptimizationNodes: Definition[] = [
  {
    id: "opt-kv-cache",
    name: "KV cache optimization",
    kind: "optimization",
    category: "Runtime tuning",
    supported: true,
    inputs: { input: "runtime" },
    output: "runtime",
    description:
      "Configure reuse of attention key/value states in the synthetic serving profile.",
    fields: [
      {
        key: "cache",
        label: "KV cache",
        default: "Enabled",
        options: ["Enabled", "Disabled"],
        help: "Enabled illustrates faster repeated decoding with additional cache memory. No real GPU cache is allocated.",
      },
    ],
  },
  {
    id: "opt-batching",
    name: "Batch size optimization",
    kind: "optimization",
    category: "Runtime tuning",
    supported: true,
    inputs: { input: "runtime" },
    output: "runtime",
    description:
      "Tune the serving batch size to explore synthetic throughput and memory tradeoffs.",
    fields: [
      number(
        "batch",
        "Batch size",
        4,
        1,
        32,
        "Default: 4 requests. Useful capacity is limited by workload concurrency; larger batches can increase the memory estimate.",
      ),
    ],
  },
  {
    id: "opt-context",
    name: "Context budget optimization",
    kind: "optimization",
    category: "Runtime tuning",
    supported: true,
    inputs: { input: "runtime" },
    output: "runtime",
    description:
      "Set the input-context budget used by the synthetic runtime memory estimate.",
    fields: [
      number(
        "context",
        "Context budget",
        2048,
        256,
        32768,
        "Default: 2,048 input tokens. Benchmark input length must fit this budget; the fixture does not model combined input/output cache occupancy.",
      ),
    ],
  },
];
export const optimizationNodeIds = new Set(
  defaultOptimizationNodes.map((n) => n.id),
);
export const performanceNodes: Definition[] = [
  ...defaultOptimizationNodes,
  {
    id: "runtime-config",
    name: "Runtime configuration",
    kind: "optimization",
    category: "Inference optimization",
    supported: true,
    inputs: { input: "model" },
    output: "runtime",
    description:
      "Describe a serving configuration without modifying model weights.",
    fields: [
      {
        key: "cache",
        label: "KV cache",
        default: "Enabled",
        options: ["Enabled", "Disabled"],
        help: "Illustrates a cache-related throughput and memory tradeoff; no runtime is configured.",
      },
      number(
        "batch",
        "Batch size",
        4,
        1,
        32,
        "Requests grouped in one synthetic serving step.",
      ),
      number(
        "context",
        "Context budget",
        2048,
        256,
        32768,
        "Illustrative maximum input tokens per request.",
      ),
    ],
  },
  {
    id: "workload",
    name: "Benchmark workload",
    kind: "validation",
    category: "Performance validation",
    supported: true,
    inputs: { input: "runtime" },
    output: "workload",
    description:
      "Define the request volume and token lengths used for a repeatable synthetic comparison.",
    fields: [
      number(
        "requests",
        "Request count",
        100,
        1,
        10000,
        "Number of requests represented in the fixture; no requests are sent.",
      ),
      number(
        "inputTokens",
        "Input tokens",
        512,
        1,
        32768,
        "Input length must fit within the runtime context budget.",
      ),
      number(
        "outputTokens",
        "Output tokens",
        128,
        1,
        4096,
        "Synthetic output length per request.",
      ),
      number(
        "concurrency",
        "Concurrency",
        4,
        1,
        64,
        "Simultaneous requests represented by the fixture.",
      ),
    ],
  },
  {
    id: "benchmark",
    name: "Performance benchmark",
    kind: "validation",
    category: "Performance validation",
    supported: true,
    inputs: { input: "workload" },
    output: "metrics",
    description:
      "Calculate deterministic illustrative latency, throughput, and memory figures for the configured workload.",
    fields: [],
  },
  {
    id: "optimization-report",
    name: "Optimization comparison",
    kind: "optimization",
    category: "Inference optimization",
    supported: true,
    inputs: { input: "metrics" },
    output: "report",
    description:
      "Compare configured serving settings against a batch-one, cache-disabled fixture using the same workload.",
    fields: [],
  },
  {
    id: "validation-gate",
    name: "Performance acceptance gate",
    kind: "validation",
    category: "Performance validation",
    supported: true,
    inputs: { input: "metrics" },
    output: "report",
    description:
      "Check synthetic benchmark metrics against explicit performance thresholds and report each pass or fail.",
    fields: [
      number(
        "maxLatency",
        "Maximum p95 latency (ms)",
        3000,
        1,
        120000,
        "Pass when the illustrative p95 request latency is at or below this threshold.",
      ),
      number(
        "minThroughput",
        "Minimum throughput (tokens/s)",
        100,
        1,
        100000,
        "Pass when aggregate illustrative output throughput meets this threshold.",
      ),
      number(
        "maxMemory",
        "Maximum memory (GB)",
        8,
        1,
        192,
        "Pass when the illustrative memory estimate is within this budget.",
      ),
    ],
  },
];
export const performanceHelp: Record<
  string,
  {
    explanation: string;
    usage: string;
    considerations: string;
    workflow: string;
  }
> = {
  "opt-kv-cache": {
    explanation:
      "Overrides the KV-cache setting of the incoming runtime profile. Enabled is the default and illustrates reusing attention states during decoding, with a synthetic memory cost.",
    usage:
      "Compare enabled and disabled cache settings while keeping the model, prompt lengths, batch size, and concurrency fixed.",
    considerations:
      "This node does not allocate a real cache or change model weights. The fixture assumes a fixed throughput factor rather than measuring an actual serving engine. A later cache node overrides an earlier one.",
    workflow:
      "Runtime configuration → KV cache optimization → other runtime tuning nodes → Benchmark workload.",
  },
  "opt-batching": {
    explanation:
      "Overrides the request batch size in the incoming runtime profile. The default is four requests, and the fixture limits useful parallel capacity to the workload’s concurrency.",
    usage:
      "Compare batch sizes one and four with the same workload to explore the synthetic throughput gain and additional memory estimate.",
    considerations:
      "The fixture does not implement request scheduling or continuous batching. Real latency depends on traffic and hardware. Later batch-size nodes override earlier settings.",
    workflow:
      "Place after Runtime configuration or another runtime tuning node, before Benchmark workload.",
  },
  "opt-context": {
    explanation:
      "Overrides the runtime’s input-context budget, defaulting to 2,048 tokens. The budget influences the synthetic memory estimate and is checked against benchmark input length.",
    usage:
      "Reduce an oversized input budget while keeping it large enough for your benchmark prompts, then compare the illustrative memory estimates.",
    considerations:
      "This does not extend a model’s trained context window, truncate prompts, or change positional embeddings. The prototype validates input length only; real backends must also account for generated tokens. Later context nodes override earlier settings.",
    workflow:
      "Runtime configuration → Context budget optimization → Benchmark workload → Performance benchmark.",
  },
  "runtime-config": {
    explanation:
      "Defines a synthetic serving profile with batch size, context budget, and KV caching. These are inference-time choices; the model reference and its weights remain unchanged.",
    usage:
      "Compare serving profiles for a sample model or a custom model exported from surgery, keeping the benchmark workload fixed.",
    considerations:
      "No serving backend is connected. Cache and batching benefits are simplified assumptions, not recommendations or measurements for your hardware.",
    workflow:
      "Model → Runtime configuration → Benchmark workload → Performance benchmark.",
  },
  workload: {
    explanation:
      "Defines input and output lengths, request count, and concurrency for a reproducible synthetic load. The same workload feeds the configured and baseline calculations.",
    usage:
      "Represent short interactive requests or longer completions, then observe how the fixture responds to different concurrency levels.",
    considerations:
      "No dataset is loaded and no traffic is generated. Request count affects estimated total duration, not the timing of the UI simulation. Inputs exceeding the context budget are rejected.",
    workflow:
      "Connect a runtime profile, configure the workload, and send it to Performance benchmark.",
  },
  benchmark: {
    explanation:
      "Produces deterministic synthetic p95 request latency, aggregate output tokens per second, peak memory, and total run duration. It also calculates a baseline with batch size one and caching disabled.",
    usage:
      "Supply a common metric set to an optimization comparison or a performance acceptance gate.",
    considerations:
      "These formulas are educational fixtures. p95 is a constructed indicator, not a percentile calculated from measured requests; no GPU, clock, or memory instrumentation runs.",
    workflow:
      "Benchmark workload → Performance benchmark → Optimization comparison or Performance acceptance gate.",
  },
  "optimization-report": {
    explanation:
      "Shows configured and baseline fixture values side by side, with percentage differences for throughput, latency, and memory. The model and workload are held constant.",
    usage:
      "Change batch size or caching and rerun to inspect the synthetic tradeoff between capacity and memory.",
    considerations:
      "A favorable simulated delta does not demonstrate a real speedup. Benchmark an actual serving runtime before applying these choices to deployment.",
    workflow:
      "Connect metrics from Performance benchmark and read the comparison in Run output.",
  },
  "validation-gate": {
    explanation:
      "Evaluates three explicit acceptance criteria: maximum p95 request latency, minimum aggregate output throughput, and maximum memory. All criteria must pass for the overall result to pass.",
    usage:
      "Specify a performance budget and identify exactly which synthetic metric misses its threshold before revising the serving profile.",
    considerations:
      "A simulated PASS is not deployment approval. This gate does not evaluate answer quality, correctness, capability retention, or actual hardware behavior.",
    workflow:
      "Connect benchmark metrics, set acceptance thresholds, and inspect the per-criterion PASS or FAIL report.",
  },
};
export type Metrics = {
  latency: number;
  throughput: number;
  memory: number;
  duration: number;
};
export function syntheticMetrics(profile: any, baseline = false): Metrics {
  const { model, settings, load } = profile;
  const batch = baseline ? 1 : Number(settings.batch),
    cache = !baseline && settings.cache === "Enabled";
  const capacity = Math.min(batch, Number(load.concurrency));
  const throughput =
    (80 * capacity * (cache ? 1.25 : 1)) /
    (1 + Number(load.inputTokens) / 4096);
  const latency =
    ((Number(load.outputTokens) / (throughput / capacity)) * 1000 +
      Number(load.inputTokens) * 0.15) *
    1.2 *
    (1 + Math.max(0, Number(load.concurrency) - batch) * 0.04);
  const memory =
    model.sizeGB +
    0.3 +
    ((batch * Number(settings.context)) / 8192) * (cache ? 1 : 0.25);
  return {
    latency,
    throughput,
    memory,
    duration: (Number(load.requests) * Number(load.outputTokens)) / throughput,
  };
}
export function comparisonReport(value: {
  configured: Metrics;
  baseline: Metrics;
}): string {
  const { configured: c, baseline: b } = value;
  const delta = (x: number, y: number) => `${((x / y - 1) * 100).toFixed(1)}%`;
  return `SYNTHETIC PERFORMANCE COMPARISON · NOT MEASURED\nSame model and workload; baseline batch = 1, cache = disabled.\n\nMetric                       Baseline → Configured\np95 request latency (ms): ${b.latency.toFixed(0)} → ${c.latency.toFixed(0)} (${delta(c.latency, b.latency)})\nOutput throughput (tokens/s): ${b.throughput.toFixed(1)} → ${c.throughput.toFixed(1)} (${delta(c.throughput, b.throughput)})\nMemory estimate (GB): ${b.memory.toFixed(2)} → ${c.memory.toFixed(2)} (${delta(c.memory, b.memory)})\nTotal workload duration (s): ${b.duration.toFixed(1)} → ${c.duration.toFixed(1)}\n\nIllustrative formulas only. Validate on a real serving backend before deployment.`;
}
export function acceptanceReport(
  m: Metrics,
  c: Record<string, string | number>,
): string {
  const checks = [
    {
      name: "p95 latency",
      actual: m.latency,
      max: Number(c.maxLatency),
      unit: "ms",
      pass: m.latency <= Number(c.maxLatency),
    },
    {
      name: "Output throughput",
      actual: m.throughput,
      max: Number(c.minThroughput),
      unit: "tokens/s",
      pass: m.throughput >= Number(c.minThroughput),
    },
    {
      name: "Memory",
      actual: m.memory,
      max: Number(c.maxMemory),
      unit: "GB",
      pass: m.memory <= Number(c.maxMemory),
    },
  ];
  return `SYNTHETIC VALIDATION · ${checks.every((x) => x.pass) ? "PASS" : "FAIL"}\nNot measured. This is not deployment approval.\n\n${checks.map((x, i) => `${x.pass ? "PASS" : "FAIL"} · ${x.name}: ${x.actual.toFixed(2)} ${x.unit} ${i === 1 ? ">=" : "<="} ${x.max} required`).join("\n")}\n\nQuality and correctness are not evaluated by this performance gate.`;
}
