import { projectedToolFields } from "./tool-fields";
import { customCodeDefinition, customDefinition } from "./custom-nodes";
import type { NodeData } from "./types";
import { surgeryOperations } from "./surgery-operations";
import { performanceNodes } from "./performance";
import raw from "./catalog.json";
import type { Definition, Field } from "./types";
const f = (
  key: string,
  label: string,
  value: string | number,
  help: string,
  min?: number,
  max?: number,
  step?: number,
): Field => ({ key, label, default: value, help, min, max, step });
export const samplingFields = [
  f(
    "temperature",
    "Temperature",
    0.7,
    "Lower values favor predictable candidates; higher values add variety.",
    0,
    2,
    0.1,
  ),
  f(
    "topK",
    "Top-K",
    40,
    "Keep only the K highest-ranked fixture candidates.",
    1,
    100,
    1,
  ),
  f(
    "topP",
    "Top-P",
    0.9,
    "Restrict the cumulative probability mass of candidate tokens.",
    0.01,
    1,
    0.01,
  ),
  f(
    "seed",
    "Random seed",
    42,
    "The same workflow and seed produce the same simulated result.",
    0,
    2147483647,
    1,
  ),
  f(
    "repetition",
    "Repetition penalty",
    1.1,
    "Discourage repeated words in the simulated output.",
    1,
    2,
    0.05,
  ),
  f(
    "maxTokens",
    "Maximum output tokens",
    80,
    "Limit the fixture output. The prototype uses whitespace tokens.",
    1,
    512,
    1,
  ),
  f(
    "stop",
    "Stop sequences",
    "",
    "Comma-separated strings that end the simulated completion.",
  ),
];
const modelFields = [
  f(
    "name",
    "Model name",
    "Kykeon Mini 1B",
    "A sample model record; no checkpoint is loaded.",
  ),
  f(
    "path",
    "Local path reference",
    "/models/kykeon-mini-1b",
    "Reference only. This browser does not read the path.",
  ),
  f(
    "architecture",
    "Architecture",
    "decoder-1b",
    "Merging requires matching architecture identifiers.",
  ),
];
export const catalog: Definition[] = raw.map(
  (item) =>
    ({
      ...item,
      supported: false,
      inputs: item.kind === "model" ? {} : { input: "model" },
      output: "model",
      fields: [],
    }) as Definition,
);
function enable(id: string, changes: Partial<Definition>) {
  Object.assign(
    catalog.find((x) => x.id === id)!,
    { supported: true },
    changes,
  );
}
enable("model-2", { fields: modelFields });
enable("surgery-25", {
  inputs: { a: "model", b: "model" },
  fields: [
    f(
      "weight",
      "Source A weight",
      0.5,
      "Source B receives 1 − A. Only matching architectures may merge.",
      0,
      1,
      0.05,
    ),
  ],
});
enable("surgery-11", {
  inputs: { input: "model" },
  fields: [
    {
      ...f(
        "precision",
        "Precision",
        "INT4",
        "Illustrative weight-size scaling, not a measured benchmark.",
      ),
      options: ["FP16", "INT8", "INT4"],
    },
  ],
});
const controlMap: Record<string, number> = {
  "control-1": 0,
  "control-2": 1,
  "control-3": 2,
  "control-10": 3,
  "control-11": 4,
  "control-21": 5,
  "control-23": 6,
};
for (const [id, index] of Object.entries(controlMap))
  enable(id, {
    inputs: { input: "context" },
    output: "context",
    fields: [samplingFields[index]],
  });
for (const entry of catalog) {
  if (entry.kind === "surgery")
    entry.fields.push(...projectedToolFields(entry.id));
}
export const utilities: Definition[] = [
  {
    id: "prompt",
    name: "Prompt",
    description:
      "Give the model a task and inspect how context enters the workflow.",
    category: "Workflow essentials",
    kind: "utility",
    supported: true,
    inputs: { input: "model" },
    output: "context",
    fields: [
      f(
        "prompt",
        "Your prompt",
        "Explain how a language model generates text.",
        "The prompt selects an educational fixture: models, merging, or quantization. Other prompts receive a general illustrative response.",
      ),
    ],
  },
  {
    id: "sampling",
    name: "Sampling controls",
    description:
      "Shape the candidate distribution without changing model weights.",
    category: "Workflow essentials",
    kind: "control",
    supported: true,
    inputs: { input: "context" },
    output: "context",
    fields: samplingFields,
  },
  {
    id: "generate",
    name: "Generate text",
    description:
      "Run a deterministic educational simulation. No language model is executed.",
    category: "Workflow essentials",
    kind: "utility",
    supported: true,
    inputs: { input: "context" },
    output: "generation",
    fields: [],
  },
  {
    id: "compare",
    name: "Output comparison",
    description:
      "Compare a default fixture with the configured simulated response.",
    category: "Workflow essentials",
    kind: "utility",
    supported: true,
    inputs: { input: "generation" },
    output: "report",
    fields: [],
  },
  {
    id: "evaluate",
    name: "Evaluate model",
    description:
      "Compare illustrative size and quality indicators against a baseline.",
    category: "Workflow essentials",
    kind: "utility",
    supported: true,
    inputs: { input: "model" },
    output: "model",
    fields: [],
  },
  {
    id: "export",
    name: "Custom model export",
    description:
      "Prepare a simulated model manifest to download and reuse in Inference.",
    category: "Workflow essentials",
    kind: "utility",
    supported: true,
    inputs: { input: "model" },
    output: "model",
    fields: [],
  },
  {
    id: "custom",
    name: "Custom model",
    description: "A simulated model exported from your surgery workflow.",
    category: "Your models",
    kind: "model",
    supported: true,
    inputs: {},
    output: "model",
    fields: [],
  },
];
export const definitions = [
  ...catalog,
  ...utilities,
  ...performanceNodes,
  ...surgeryOperations,
  customCodeDefinition,
];
export function definition(id: string): Definition {
  const d = definitions.find((x) => x.id === id);
  if (!d) throw Error(`Unknown node type: ${id}`);
  return d;
}

export function nodeDefinition(data: NodeData): Definition {
  return data.definitionId === "custom-code"
    ? customDefinition(data)
    : definition(data.definitionId);
}
