import { validateCustomConfig, sourceTemplates } from "./custom-nodes";
import { describe, it, expect } from "vitest";
import { catalog, definitions } from "./catalog";
import { nodeHelp } from "./node-help";
import {
  starter,
  createNode,
  connectionError,
  parseWorkflow,
  validateWorkflow,
  simulationEngine,
  simulateText,
  historyPush,
  parseManifests,
} from "./engine";
const settings = {
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  seed: 42,
  repetition: 1.1,
  maxTokens: 80,
  stop: "",
};
describe("source catalog", () => {
  it("retains all 150 source entries and 10 model categories", () => {
    for (const kind of ["model", "control", "surgery"])
      expect(catalog.filter((d) => d.kind === kind)).toHaveLength(50);
    expect(
      new Set(catalog.filter((d) => d.kind === "model").map((d) => d.category))
        .size,
    ).toBe(10);
    expect(new Set(catalog.map((d) => d.id)).size).toBe(150);
  });
});
describe("workflow safety", () => {
  it("round trips every starter", () => {
    for (const k of ["inference", "merge", "quantize"] as const) {
      const w = starter(k);
      expect(parseWorkflow(JSON.stringify(w))).toEqual(w);
      expect(() => validateWorkflow(w)).not.toThrow();
    }
  });
  it("rejects incompatible and occupied inputs", () => {
    const w = starter("inference");
    expect(connectionError(w, w.nodes[0].id, w.nodes[3].id, "input")).toMatch(
      /incompatible/,
    );
    expect(connectionError(w, w.nodes[0].id, w.nodes[1].id, "input")).toMatch(
      /already/,
    );
  });
  it("rejects cycles", () => {
    const a = createNode("sampling"),
      b = createNode("sampling");
    const w = {
      ...starter("inference"),
      nodes: [a, b],
      edges: [
        {
          id: "ab",
          source: a.id,
          target: b.id,
          sourceHandle: "out",
          targetHandle: "input",
        },
      ],
    };
    expect(connectionError(w, b.id, a.id, "input")).toMatch(/cycle/);
  });
  it("blocks catalog-only nodes and disconnected inputs", () => {
    const w = starter("inference");
    w.nodes.push(createNode("model-1"));
    expect(() => validateWorkflow(w)).toThrow(/catalog-only/);
    w.nodes.pop();
    w.edges = [];
    expect(() => validateWorkflow(w)).toThrow(/connect/);
  });
  it("rejects malformed imports without mutating the original", () => {
    const w = starter("merge"),
      before = JSON.stringify(w);
    expect(() => parseWorkflow("{")).toThrow();
    expect(() => parseWorkflow(JSON.stringify({ ...w, version: 2 }))).toThrow(
      /version/,
    );
    expect(() =>
      parseWorkflow(JSON.stringify({ ...w, nodes: [w.nodes[0], w.nodes[0]] })),
    ).toThrow(/duplicate/);
    expect(JSON.stringify(w)).toBe(before);
  });
  it("rejects out-of-range run configuration", () => {
    const w = starter("inference");
    w.nodes[2].data.config.temperature = -1;
    expect(() => validateWorkflow(w)).toThrow(/range/);
  });
  it("preserves historical snapshots for undo and redo", () => {
    const a = starter("inference");
    const b = { ...a, name: "Changed" };
    const past = historyPush([], a);
    const future = [b];
    expect(past.at(-1)?.name).toBe(a.name);
    expect(future[0].name).toBe("Changed");
    expect(historyPush(Array(50).fill(a), b)).toHaveLength(50);
  });
});
describe("simulations", () => {
  it("is deterministic and respects length and stop controls", () => {
    const p = "Explain language models";
    expect(simulateText(p, settings)).toBe(simulateText(p, settings));
    expect(
      simulateText(p, { ...settings, maxTokens: 3 }).split(" "),
    ).toHaveLength(3);
    expect(simulateText(p, { ...settings, stop: "predicts" })).not.toContain(
      "predicts",
    );
    expect(
      simulateText(p, { ...settings, repetition: 1 }).length,
    ).toBeGreaterThan(simulateText(p, settings).length);
  });
  it("completes inference with observable progress", async () => {
    const events: string[] = [];
    const r = await simulationEngine.run(
      starter("inference"),
      (e) => events.push(e.state),
      new AbortController().signal,
    );
    expect(r.text).toContain("language model");
    expect(r.report).toContain("Default fixture");
    expect(events.filter((e) => e === "complete")).toHaveLength(5);
  });
  it("cancels promptly", async () => {
    const controller = new AbortController();
    const promise = simulationEngine.run(
      starter("merge"),
      () => controller.abort(),
      controller.signal,
    );
    await expect(promise).rejects.toThrow(/cancelled/);
  });
  it("exports merge lineage and supports custom-model reuse", async () => {
    const w = starter("merge");
    const r = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    expect(r.model?.sources).toHaveLength(2);
    expect(r.model?.history[0]).toContain("Linear merge");
    const manifest = {
      version: 1,
      simulated: true,
      id: "test",
      model: r.model,
    };
    expect(parseManifests([manifest])).toHaveLength(1);
    const inference = starter("inference");
    inference.nodes[0].data = {
      definitionId: "custom",
      config: {},
      model: r.model,
    };
    const reused = await simulationEngine.run(
      inference,
      () => {},
      new AbortController().signal,
    );
    expect(reused.text).toBeTruthy();
  }, 10000);
  it("rejects incompatible merge architectures", async () => {
    const w = starter("merge");
    w.nodes[1].data.config.architecture = "other";
    await expect(
      simulationEngine.run(w, () => {}, new AbortController().signal),
    ).rejects.toThrow(/matching/);
  });
  it("quantizes without mutating source and labels estimates", async () => {
    const w = starter("quantize");
    const before = JSON.stringify(w);
    const r = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    expect(r.model?.sizeGB).toBe(0.5);
    expect(r.report).toContain("not measured");
    expect(JSON.stringify(w)).toBe(before);
  });
});

describe("selected-node documentation", () => {
  it("provides detailed help and a unique practical use for every catalog node", () => {
    const uses = new Set<string>();
    for (const entry of definitions) {
      const help = nodeHelp(entry);
      expect(help.explanation.length).toBeGreaterThan(60);
      expect(help.usage.length).toBeGreaterThan(40);
      expect(help.considerations.length).toBeGreaterThan(50);
      expect(help.workflow.length).toBeGreaterThan(25);
      uses.add(help.usage);
    }
    expect(uses.size).toBe(definitions.length);
  });
});

describe("performance workspaces", () => {
  it("round trips both new modes with correctly typed starter connections", () => {
    for (const kind of ["optimization", "validation"] as const) {
      const w = starter(kind);
      expect(w.mode).toBe(kind);
      expect(parseWorkflow(JSON.stringify(w))).toEqual(w);
      expect(() => validateWorkflow(w)).not.toThrow();
    }
  });
  it("compares configurations without modifying source weights", async () => {
    const w = starter("optimization");
    const before = JSON.stringify(w);
    const r = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    expect(r.report).toContain("NOT MEASURED");
    expect(r.report).toContain("Baseline → Configured");
    expect(r.model).toBeUndefined();
    expect(JSON.stringify(w)).toBe(before);
  });
  it("evaluates both passing and failing acceptance thresholds", async () => {
    const w = starter("validation");
    const pass = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    expect(pass.report).toContain("VALIDATION · PASS");
    w.nodes[4].data.config.maxLatency = 1;
    const fail = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    expect(fail.report).toContain("VALIDATION · FAIL");
    expect(fail.report).toContain("FAIL · p95 latency");
  }, 10000);
  it("rejects a workload exceeding the context budget", async () => {
    const w = starter("optimization");
    w.nodes.find(
      (n) => n.data.definitionId === "workload",
    )!.data.config.inputTokens = 4096;
    await expect(
      simulationEngine.run(w, () => {}, new AbortController().signal),
    ).rejects.toThrow(/context budget/);
  });
});

describe("surgery operation resource flags", () => {
  it("keeps the 50 operations separate from tool names and classifies scope explicitly", () => {
    const ops = definitions.filter((d) => d.kind === "operation");
    expect(ops).toHaveLength(50);
    expect(new Set(ops.map((d) => d.category)).size).toBe(10);
    expect(
      ops
        .filter((d) => d.resourceProfile?.tier === "enterprise")
        .map((d) => d.resourceProfile?.level),
    ).toEqual([48, 49, 50]);
    expect(
      ops.find((d) => d.id === "operation-41")?.resourceProfile?.tier,
    ).toBe("recipe-dependent");
    expect(
      ops.every(
        (d) => !d.supported && d.resourceProfile?.basis === "provisional-scope",
      ),
    ).toBe(true);
    expect(
      definitions.find((d) => d.id === "surgery-25")?.resourceProfile,
    ).toBeUndefined();
  });
  it("preserves operation references through export/import without granting execution", () => {
    const w = starter("merge");
    w.nodes.push(createNode("operation-49"));
    const restored = parseWorkflow(JSON.stringify(w));
    expect(restored.nodes.at(-1)?.data.definitionId).toBe("operation-49");
    expect(() => validateWorkflow(restored)).toThrow(/catalog-only/);
  });
  it("fills weight-rescaling fields without turning a saved workflow into local execution", () => {
    const w = starter("inference");
    const node = createNode("operation-4");
    node.data.config = {};
    w.nodes.push(node);
    const raw = JSON.parse(JSON.stringify(w));
    raw.execution = "local";
    const restored = parseWorkflow(JSON.stringify(raw));
    expect(restored.nodes.at(-1)?.data.config.scale).toBe(1);
    expect(restored.nodes.at(-1)?.data.config.tensor).toBe("");
    expect(restored).not.toHaveProperty("execution");
    expect(restored.version).toBe(1);
    expect(() => validateWorkflow(restored)).toThrow(/no simulation adapter/);
    const rescale = starter("rescale");
    expect(parseWorkflow(JSON.stringify(rescale))).toEqual(rescale);
    expect(rescale.mode).toBe("surgery");
    expect(() => validateWorkflow(rescale)).toThrow(/no simulation adapter/);
  });
});

describe("default optimization nodes", () => {
  it("applies connected tuning overrides without changing the source runtime", async () => {
    const w = starter("optimization");
    const runtime = w.nodes.find(
      (n) => n.data.definitionId === "runtime-config",
    )!;
    const cache = w.nodes.find((n) => n.data.definitionId === "opt-kv-cache")!;
    const batch = w.nodes.find((n) => n.data.definitionId === "opt-batching")!;
    const context = w.nodes.find((n) => n.data.definitionId === "opt-context")!;
    expect(cache.data.config.cache).toBe("Enabled");
    expect(batch.data.config.batch).toBe(4);
    expect(context.data.config.context).toBe(2048);
    runtime.data.config.cache = "Disabled";
    batch.data.config.batch = 1;
    context.data.config.context = 1024;
    const result = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    const before = result.values[runtime.id] as {
      settings: Record<string, unknown>;
    };
    const after = result.values[context.id] as {
      settings: Record<string, unknown>;
    };
    expect(before.settings.cache).toBe("Disabled");
    expect(before.settings.batch).toBe(4);
    expect(after.settings).toEqual({
      cache: "Enabled",
      batch: 1,
      context: 1024,
    });
    expect(result.report).toContain("NOT MEASURED");
  });
});

describe("custom source nodes", () => {
  it("preserves labeled source for all formats and workspace modes", () => {
    for (const kind of [
      "inference",
      "merge",
      "optimization",
      "validation",
    ] as const) {
      for (const format of ["JSON", "PyTorch", "TensorFlow"] as const) {
        const w = starter(kind),
          n = createNode("custom-code");
        n.data.config = {
          label: "My transform",
          format,
          source: sourceTemplates[format],
          inputType: "model",
          outputType: "model",
        };
        w.nodes.push(n);
        expect(
          parseWorkflow(JSON.stringify(w)).nodes.at(-1)?.data.config,
        ).toEqual(n.data.config);
        expect(() => validateWorkflow(w)).toThrow(
          /My transform is catalog-only/,
        );
      }
    }
  });
  it("rejects invalid source, blank labels, oversized source, unknown formats and ports", () => {
    const good = {
      label: "My custom node",
      format: "JSON",
      source: "{}",
      inputType: "model",
      outputType: "model",
    };
    for (const change of [
      { label: " " },
      { source: "{" },
      { source: "[]" },
      { source: "x".repeat(100001) },
      { format: "JavaScript" },
      { inputType: "invalid" },
    ])
      expect(() => validateCustomConfig({ ...good, ...change })).toThrow();
    const w = starter("inference"),
      n = createNode("custom-code");
    n.data.config = { ...good, source: "{" };
    w.nodes.push(n);
    expect(() => parseWorkflow(JSON.stringify(w))).toThrow(/Invalid JSON/);
  });
  it("uses the declared custom ports to validate connections", () => {
    const w = starter("optimization"),
      n = createNode("custom-code");
    n.data.config = {
      label: "Tune runtime",
      format: "PyTorch",
      source: sourceTemplates.PyTorch,
      inputType: "runtime",
      outputType: "runtime",
    };
    w.nodes.push(n);
    const runtime = w.nodes.find(
      (n) => n.data.definitionId === "runtime-config",
    )!;
    expect(connectionError(w, runtime.id, n.id, "input")).toBeNull();
    expect(connectionError(w, w.nodes[0].id, n.id, "input")).toMatch(
      /incompatible/,
    );
  });
});

describe("projected tool interfaces", () => {
  it("provides unique, documented, persistable fields for all 50 tools", () => {
    const tools = catalog.filter((d) => d.kind === "surgery");
    expect(tools).toHaveLength(50);
    for (const tool of tools) {
      expect(
        tool.fields.filter((f) => f.projected).length,
      ).toBeGreaterThanOrEqual(4);
      expect(new Set(tool.fields.map((f) => f.key)).size).toBe(
        tool.fields.length,
      );
      expect(tool.fields.every((f) => f.help.length > 10)).toBe(true);
      const node = createNode(tool.id);
      const w = { ...starter("merge"), nodes: [node], edges: [] };
      node.data.config.planned_outputReference = "/outputs/custom-artifact";
      expect(parseWorkflow(JSON.stringify(w)).nodes[0].data.config).toEqual(
        node.data.config,
      );
    }
  });
  it("upgrades legacy nodes while preserving their original demo controls", () => {
    const w = starter("merge");
    for (const node of w.nodes)
      for (const key of Object.keys(node.data.config))
        if (key.startsWith("planned_")) delete node.data.config[key];
    w.nodes[2].data.config.weight = 0.7;
    const restored = parseWorkflow(JSON.stringify(w));
    expect(restored.nodes[2].data.config.weight).toBe(0.7);
    expect(restored.nodes[2].data.config.planned_mergeMethod).toBe("Linear");
    expect(() => validateWorkflow(restored)).not.toThrow();
  });
  it("rejects malformed projected values on import", () => {
    const w = starter("merge");
    w.nodes[2].data.config.planned_mergeMethod = "Imaginary method";
    expect(() => parseWorkflow(JSON.stringify(w))).toThrow(/projected field/);
  });
  it("keeps projected method selections from changing the simulated merge", async () => {
    const w = starter("merge");
    w.nodes[2].data.config.planned_mergeMethod = "SLERP";
    const r = await simulationEngine.run(
      w,
      () => {},
      new AbortController().signal,
    );
    expect(r.model?.history[0]).toContain("Linear merge");
  });
});
