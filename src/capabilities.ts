import type { Definition } from "./types";

const realLocalNodes = new Set([
  "model-2",
  "prompt",
  "sampling",
  "generate",
  "compare",
  "control-1",
  "control-2",
  "control-3",
  "control-10",
  "control-11",
  "control-21",
  "control-23",
  "operation-4",
  "evaluate",
  "export",
]);

export function capabilityLabels(definition: Definition): string[] {
  const labels = ["Catalog reference"];
  if (definition.supported) labels.push("Simulation available");
  if (realLocalNodes.has(definition.id))
    labels.push("Real local execution available");
  if (definition.id === "operation-4") labels.push("Unsupported in simulation");
  else if (!definition.supported) labels.push("Unsupported configuration");
  if (definition.resourceProfile?.tier === "enterprise")
    labels.push("Enterprise classification");
  else if (definition.resourceProfile?.tier === "recipe-dependent")
    labels.push("Recipe-dependent planning classification");
  else if (definition.resourceProfile?.tier === "local-candidate")
    labels.push("Local-candidate planning classification");
  return labels;
}
