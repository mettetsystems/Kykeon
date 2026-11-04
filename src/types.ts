import type { Node, Edge } from "@xyflow/react";
export type Port =
  | "model"
  | "context"
  | "generation"
  | "report"
  | "runtime"
  | "workload"
  | "metrics";
export type Field = {
  projected?: boolean;
  multiline?: boolean;
  key: string;
  label: string;
  help: string;
  default: string | number;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
};
export type ResourceProfile = {
  tier: "enterprise" | "recipe-dependent" | "local-candidate";
  reason: string;
  level: number;
  basis: "provisional-scope";
};
export type Definition = {
  resourceProfile?: ResourceProfile;
  id: string;
  name: string;
  description: string;
  category: string;
  kind: string;
  supported: boolean;
  inputs: Record<string, Port>;
  output: Port;
  fields: Field[];
};
export type ModelReference = {
  name: string;
  path: string;
  architecture: string;
  sizeGB: number;
  history: string[];
  sources: string[];
};
export type NodeData = {
  definitionId: string;
  config: Record<string, string | number>;
  model?: ModelReference;
  [key: string]: unknown;
};
export type FlowNode = Node<NodeData>;
export type Workflow = {
  version: 1;
  name: string;
  mode: "inference" | "surgery" | "optimization" | "validation";
  nodes: FlowNode[];
  edges: Edge[];
};
export type CustomModelManifest = {
  version: 1;
  simulated: true;
  id: string;
  model: ModelReference;
};
export type RunEvent = {
  nodeId: string;
  state: "running" | "complete";
  detail: string;
};
export type RunResult = {
  text?: string;
  model?: ModelReference;
  report?: string;
  values: Record<string, unknown>;
};
export interface ExecutionEngine {
  run(
    workflow: Workflow,
    onEvent: (event: RunEvent) => void,
    signal: AbortSignal,
  ): Promise<RunResult>;
}
