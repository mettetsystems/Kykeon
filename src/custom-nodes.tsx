import { useEffect, useRef, useState } from "react";
import type { Definition, NodeData, Port } from "./types";
export const customPorts: Port[] = [
  "model",
  "context",
  "generation",
  "report",
  "runtime",
  "workload",
  "metrics",
];
export const customCodeDefinition: Definition = {
  id: "custom-code",
  name: "Custom code node",
  description:
    "User-provided JSON or Python source, stored for a future execution backend.",
  category: "Custom nodes",
  kind: "custom-code",
  supported: false,
  inputs: { input: "model" },
  output: "model",
  fields: [],
};
export const sourceTemplates = {
  JSON: '{\n  "operation": "my_custom_operation",\n  "parameters": {}\n}',
  PyTorch:
    "import torch\n\ndef transform(value):\n    # Implement your operation here.\n    return value\n",
  TensorFlow:
    "import tensorflow as tf\n\ndef transform(value):\n    # Implement your operation here.\n    return value\n",
};
export function validateCustomConfig(config: Record<string, unknown>) {
  if (
    typeof config.label !== "string" ||
    !config.label.trim() ||
    config.label.length > 80
  )
    throw Error("Enter a node label between 1 and 80 characters.");
  if (!Object.hasOwn(sourceTemplates, String(config.format)))
    throw Error("Choose JSON, PyTorch, or TensorFlow.");
  if (
    typeof config.source !== "string" ||
    !config.source.trim() ||
    config.source.length > 100000
  )
    throw Error("Source must contain 1–100,000 characters.");
  if (
    !customPorts.includes(config.inputType as Port) ||
    !customPorts.includes(config.outputType as Port)
  )
    throw Error("Choose valid input and output types.");
  if (config.format === "JSON") {
    let value;
    try {
      value = JSON.parse(config.source);
    } catch {
      throw Error("Invalid JSON. Check the syntax before saving.");
    }
    if (!value || typeof value !== "object" || Array.isArray(value))
      throw Error(
        "JSON must be an object containing your custom configuration.",
      );
  }
}
export function customDefinition(data: NodeData): Definition {
  return {
    ...customCodeDefinition,
    name: String(data.config.label || "Custom code node"),
    inputs: { input: data.config.inputType as Port },
    output: data.config.outputType as Port,
  };
}
export function CustomNodeDialog({
  mode,
  initial,
  onSave,
  onClose,
}: {
  mode: string;
  initial?: NodeData["config"];
  onSave: (config: NodeData["config"]) => void;
  onClose: () => void;
}) {
  const [config, setConfig] = useState<NodeData["config"]>(
    initial || {
      label: "",
      format: "JSON",
      source: sourceTemplates.JSON,
      inputType: mode === "optimization" ? "runtime" : "model",
      outputType: mode === "optimization" ? "runtime" : "model",
    },
  );
  const [error, setError] = useState("");
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  function set(key: string, value: string) {
    setConfig((c) => ({ ...c, [key]: value }));
    setError("");
  }
  return (
    <dialog
      ref={ref}
      className="custom-dialog"
      onCancel={onClose}
      onClose={onClose}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          try {
            validateCustomConfig(config);
            onSave({ ...config, label: String(config.label).trim() });
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      >
        <div className="dialog-heading">
          <h2>{initial ? "Edit custom node" : "Add custom node"}</h2>
          <button
            type="button"
            aria-label="Close custom node editor"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p>
          Label your node and paste or import its source. Source is saved with
          the workflow; Python and custom JSON operations do not execute in this
          prototype.
        </p>
        <label className="field">
          <span>Node label</span>
          <input
            autoFocus
            required
            maxLength={80}
            value={config.label}
            onChange={(e) => set("label", e.target.value)}
            placeholder="e.g. My attention transform"
          />
        </label>
        <label className="field">
          <span>Source format</span>
          <select
            value={config.format}
            onChange={(e) => set("format", e.target.value)}
          >
            {Object.keys(sourceTemplates).map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </label>
        <div className="dialog-ports">
          {["inputType", "outputType"].map((key) => (
            <label className="field" key={key}>
              <span>{key === "inputType" ? "Input type" : "Output type"}</span>
              <select
                value={config[key]}
                onChange={(e) => set(key, e.target.value)}
              >
                {customPorts.map((p) => (
                  <option key={p}>{p}</option>
                ))}
              </select>
            </label>
          ))}
        </div>
        <label className="field">
          <span>{config.format} source</span>
          <textarea
            className="source-editor"
            spellCheck={false}
            required
            value={config.source}
            onChange={(e) => set("source", e.target.value)}
          />
        </label>
        <div className="source-tools">
          <button
            type="button"
            onClick={() =>
              set(
                "source",
                sourceTemplates[config.format as keyof typeof sourceTemplates],
              )
            }
          >
            Insert example (replaces source)
          </button>
          <label>
            Import source file
            <input
              type="file"
              accept=".json,.py,.txt"
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                if (f.size > 100000) {
                  setError("File exceeds 100 KB.");
                  return;
                }
                try {
                  set("source", await f.text());
                } catch {
                  setError("Could not read the selected file.");
                }
                e.target.value = "";
              }}
            />
          </label>
        </div>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button className="accent-button" type="submit">
            {initial ? "Save custom node" : "Add to workflow"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
