import raw from "./surgery-operations.json";
import type { Definition, Field } from "./types";
export const localScope =
  "Planning baseline: one RTX 5090, approximately 3B parameters, and adaptation no heavier than PEFT/LoRA. No hardware benchmarks have been performed.";
const localLevels = new Set([
  1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 18, 19, 21, 22, 23, 24, 25,
]);
const enterpriseReasons: Record<number, string> = {
  48: "The catalog recipe trains a substantially new architecture to reproduce internal representations. Broad student training exceeds the local PEFT-only adaptation scope; a restricted adapter-only recipe would need a separate assessment.",
  49: "The catalog recipe trains a different student architecture through distillation. Full-student adaptation is outside the local PEFT budget; small or adapter-only distillation is not automatically Enterprise.",
  50: "Predominantly training a new architecture from scratch is outside Kykeon’s local model-customization scope. This reference is Enterprise/out of scope, not a promised enterprise implementation.",
};
export const surgeryOperations: Definition[] = raw.map((row) => {
  const tier = enterpriseReasons[row.level]
    ? "enterprise"
    : localLevels.has(row.level)
      ? "local-candidate"
      : "recipe-dependent";
  const reason =
    enterpriseReasons[row.level] ||
    (tier === "local-candidate"
      ? "A bounded version of this operation is a local implementation candidate on a compatible small pretrained model. Memory use and any recovery training still require recipe-specific checks; this is not a verified execution capability."
      : "The operation name does not establish its memory or training needs. Extent of replacement, trainable parameters, sequence length, teacher residency, and recovery targets must be assessed. Escalate to Enterprise if the chosen recipe exceeds the local PEFT budget.");
  return {
    id: `operation-${row.level}`,
    name: row.name,
    description: row.description,
    category: row.category,
    kind: "operation",
    supported: false,
    inputs: { input: "model" },
    output: "model",
    fields: [],
    resourceProfile: {
      tier,
      reason,
      level: row.level,
      basis: "provisional-scope",
    },
    };
});
const field = (
  key: string,
  label: string,
  value: string | number,
  help: string,
  extra: Partial<Field> = {},
): Field => ({ key, label, default: value, help, ...extra });
const rescale = surgeryOperations.find((entry) => entry.id === "operation-4");
if (rescale) {
  rescale.fields = [
    field(
      "tensor",
      "Weight tensor",
      "",
      "Exact safetensors name from the inspected checkpoint. Nothing is selected until you choose one projection.",
    ),
    field(
      "scale",
      "Scale factor",
      1,
      "Multiplies the selected tensor. Finite values from 0.5 to 1.5. 1.0 leaves the values unchanged.",
      { min: 0.5, max: 1.5, step: 0.05 },
    ),
    field(
      "prompt",
      "Comparison prompt",
      "Explain how a language model generates text.",
      "Used for the before/after generation. Local execution tokenizes this text; simulation does not.",
      { multiline: true },
    ),
    field("temperature", "Temperature", 0.7, "0 selects greedy decoding. Above 0 samples with top-k and top-p.", {
      min: 0,
      max: 2,
      step: 0.1,
    }),
    field("topK", "Top-K", 40, "Kept only when temperature is above 0.", { min: 1, max: 100, step: 1 }),
    field("topP", "Top-P", 0.9, "Kept only when temperature is above 0.", { min: 0.01, max: 1, step: 0.01 }),
    field("seed", "Random seed", 42, "Reused for the source and derived generations.", {
      min: 0,
      max: 2147483647,
      step: 1,
    }),
    field("repetition", "Repetition penalty", 1.1, "Applied during local generation.", {
      min: 1,
      max: 2,
      step: 0.05,
    }),
    field("maxTokens", "Maximum new tokens", 80, "Counted with the checkpoint tokenizer during local execution.", {
      min: 1,
      max: 512,
      step: 1,
    }),
    field("stop", "Stop sequences", "", "Comma-separated strings. Matching works across token boundaries."),
  ];
}
export const tierLabels = {
  enterprise: "Enterprise",
  "recipe-dependent": "Recipe-dependent",
  "local-candidate": "Local candidate",
};
