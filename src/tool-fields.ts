import type { Field } from "./types";
const text = (key: string, label: string, help: string, value = ""): Field => ({
  key: `planned_${key}`,
  label,
  help,
  default: value,
  projected: true,
});
const num = (
  key: string,
  label: string,
  value: number,
  min: number,
  max: number,
  step = 1,
): Field => ({
  ...text(
    key,
    label,
    "Planning value only. Backend limits and architecture compatibility must be checked before execution.",
  ),
  default: value,
  min,
  max,
  step,
});
const choice = (key: string, label: string, options: string[]): Field => ({
  ...text(
    key,
    label,
    "Proposed Kykeon options, not a verified backend API. Availability depends on the installed tool and model.",
  ),
  default: options[0],
  options,
});
const block = (
  key: string,
  label: string,
  help: string,
  value = "",
): Field => ({ ...text(key, label, help, value), multiline: true });
const targets = () =>
  text(
    "targets",
    "Target modules / layers",
    "Comma-separated names or a selector, to be resolved against the actual model. Blank means not yet selected.",
  );
const calibration = () => [
  text(
    "calibration",
    "Calibration dataset reference",
    "Local dataset reference only. No file is opened.",
  ),
  num("samples", "Calibration samples", 128, 1, 100000),
  num("sequence", "Calibration sequence length", 512, 1, 32768),
];
const training = () => [
  text(
    "trainingData",
    "Adaptation dataset reference",
    "Required only for a recipe that trains or adapts parameters. No training occurs.",
  ),
  num("steps", "Adaptation steps", 100, 1, 1000000),
  num("learningRate", "Learning rate", 0.0001, 0.0000001, 1, 0.00001),
];
const adapters = () => [
  text(
    "adapter",
    "Adapter reference",
    "Path or identifier of a compatible adapter; reference only.",
  ),
  targets(),
  num("rank", "LoRA rank", 8, 1, 1024),
  num("alpha", "LoRA alpha", 16, 1, 4096),
  num("dropout", "LoRA dropout", 0.05, 0, 0.99, 0.01),
];
const interventions = () => [
  targets(),
  text(
    "location",
    "Activation / hook location",
    "Exact location must be supported by the model and tracing backend.",
  ),
  block(
    "testPrompt",
    "Test prompt",
    "Text used to inspect or compare the planned intervention.",
  ),
  text(
    "sourceRun",
    "Source activation reference",
    "Optional saved activation or source-run identifier; reference only.",
  ),
];
const specifics: Record<number, Field[]> = {
  1: [
    choice("structure", "Pruning structure", [
      "Channels",
      "Neurons",
      "Attention heads",
    ]),
    choice("importance", "Importance criterion", [
      "Magnitude",
      "Calibration-based",
    ]),
  ],
  2: [
    choice("scope", "Pruning scope", ["Blocks", "Channels"]),
    choice("recovery", "Recovery plan", ["Evaluate first", "PEFT adaptation"]),
  ],
  3: [
    num("sparsity", "Target weight sparsity", 0.2, 0, 0.95, 0.05),
    text(
      "pattern",
      "Sparsity pattern",
      "Unstructured or backend-supported N:M pattern.",
      "Unstructured",
    ),
  ],
  4: [
    num("sparsity", "Target weight sparsity", 0.2, 0, 0.95, 0.05),
    num("blockSize", "Processing block size", 128, 1, 4096),
  ],
  5: [
    num("reduction", "Hidden-dimension reduction", 0.1, 0, 0.9, 0.05),
    text(
      "rotation",
      "Rotation configuration reference",
      "Model-specific rotation configuration; planned only.",
    ),
  ],
  6: [
    text(
      "dropLayers",
      "Layers to remove",
      "Explicit layer indices; determine redundancy before removal.",
    ),
    text(
      "importanceData",
      "Layer-importance dataset",
      "Representative prompts for assessing layer importance.",
    ),
  ],
  7: [
    text(
      "importanceConfig",
      "Fluctuation metric configuration",
      "Reference to a model-specific importance configuration.",
    ),
  ],
  8: [
    text(
      "blocks",
      "Candidate block ranges",
      "Ranges to inspect for redundancy before removal.",
    ),
  ],
  9: [
    choice("compression", "Compression plan", [
      "Pruning",
      "Quantization",
      "Combined",
    ]),
    text(
      "preset",
      "Compression preset reference",
      "Version-specific backend configuration reference.",
    ),
  ],
  10: [
    choice("recipe", "Compression recipe", [
      "Pruning",
      "Distillation",
      "Architecture compression",
    ]),
    text(
      "teacher",
      "Teacher model reference",
      "Needed only for distillation; does not imply local feasibility.",
    ),
  ],
  11: [
    choice("quantMethod", "Quantization recipe", [
      "INT4 weight-only",
      "INT8 weight-only",
      "Custom configuration",
    ]),
    num("groupSize", "Group size", 128, 16, 1024, 16),
  ],
  12: [
    choice("loadMode", "Low-bit loading", ["4-bit", "8-bit"]),
    choice("quantType", "4-bit quantization type", ["NF4", "FP4"]),
    choice("computeDtype", "Compute dtype", ["BF16", "FP16"]),
  ],
  13: [
    num("bits", "Weight bits", 4, 2, 8),
    num("groupSize", "Group size", 128, 16, 1024, 16),
  ],
  14: [
    num("bits", "Weight bits", 4, 2, 8),
    num("groupSize", "Group size", 128, 16, 1024, 16),
  ],
  15: [
    num("bits", "Weight bits", 4, 2, 8),
    num("groupSize", "Group size", 64, 16, 1024, 16),
  ],
  16: [
    choice("weights", "Weight representation", [
      "INT8",
      "INT4",
      "Backend-specific",
    ]),
    choice("activations", "Activation representation", [
      "Unchanged",
      "INT8",
      "Backend-specific",
    ]),
  ],
  17: [
    text(
      "recipe",
      "Compression recipe reference",
      "Recipe file describing the intended compression passes.",
    ),
    text(
      "ignore",
      "Excluded modules",
      "Modules to retain at original precision.",
    ),
  ],
  18: [
    num("bits", "Weight bits", 4, 2, 8),
    num("iterations", "Rounding iterations", 200, 1, 100000),
  ],
  19: [
    choice("approach", "Compression approach", [
      "Post-training quantization",
      "Mixed precision",
    ]),
    text(
      "targetDevice",
      "Target device",
      "Deployment target to check against the installed backend.",
    ),
  ],
  20: [
    num("weightBits", "Weight bit width", 8, 2, 32),
    num("activationBits", "Activation bit width", 8, 2, 32),
  ],
  21: [
    text(
      "preset",
      "Optimization configuration reference",
      "Backend-specific preset for the chosen model and hardware.",
    ),
    text(
      "exportTarget",
      "Deployment target",
      "Target runtime or export format; compatibility is not verified.",
    ),
  ],
  22: [
    choice("ggufType", "GGUF quantization type", [
      "Q4_K_M",
      "Q5_K_M",
      "Q8_0",
      "F16",
    ]),
    text(
      "importanceMatrix",
      "Importance matrix reference",
      "Optional importance matrix used by a compatible quantization recipe.",
    ),
  ],
  23: [
    choice("optimization", "Optimization plan", [
      "Post-training quantization",
      "Pruning",
      "Clustering",
    ]),
    text(
      "converter",
      "Converter configuration",
      "TensorFlow/TFLite conversion configuration reference.",
    ),
  ],
  24: [
    choice("compression", "Compression plan", [
      "Quantization",
      "Pruning",
      "Slimming",
    ]),
    text(
      "paddleConfig",
      "Paddle configuration reference",
      "Configuration for the selected Paddle model and tool version.",
    ),
  ],
  25: [
    choice("mergeMethod", "Planned merge method", [
      "Linear",
      "SLERP",
      "TIES",
      "DARE",
      "Passthrough",
    ]),
    block(
      "slices",
      "Layer slices / weights",
      "Projected YAML or JSON describing per-source slices and parameters. The demo still performs only its simulated linear merge.",
    ),
  ],
  26: [
    choice("selection", "Soup selection", [
      "Uniform averaging",
      "Greedy selection",
    ]),
    text(
      "evaluationSet",
      "Selection evaluation set",
      "Dataset used to compare candidate soups.",
    ),
  ],
  27: [
    text(
      "recipe",
      "Diffusion merge recipe",
      "Recipe describing compatible diffusion components and weights.",
    ),
  ],
  28: [
    choice("mergeMode", "Merge mode", [
      "Weighted sum",
      "Add difference",
      "Block weights",
    ]),
    block(
      "blockWeights",
      "Block weights",
      "Proposed component-level weights; model-specific.",
    ),
  ],
  29: [
    choice("adapterAction", "Adapter operation", [
      "Attach existing adapter",
      "Merge adapter",
      "Plan LoRA training",
    ]),
    ...adapters(),
    ...training(),
  ],
  30: [
    choice("adapterAction", "LoRA operation", [
      "Merge existing LoRA",
      "Plan LoRA training",
    ]),
    ...adapters(),
    ...training(),
  ],
  31: [
    choice("adapterAction", "Adapter operation", ["Fuse LoRA", "Unfuse LoRA"]),
    text(
      "adapter",
      "LoRA reference",
      "Compatible diffusion adapter reference.",
    ),
    num("scale", "Adapter scale", 1, 0, 4, 0.1),
  ],
  32: [
    text(
      "components",
      "Checkpoint components",
      "Components to combine, such as model, text encoder, or VAE.",
    ),
    num("ratio", "Blend ratio", 0.5, 0, 1, 0.05),
  ],
  33: [
    choice("editor", "Editing method", [
      "ROME",
      "MEMIT",
      "MEND",
      "SERAC",
      "AlphaEdit",
      "GRACE",
    ]),
    text(
      "hyperparams",
      "Hyperparameter file",
      "Architecture-specific editing configuration.",
    ),
  ],
  34: [
    text(
      "rewriteModule",
      "Rewrite module",
      "Model-specific module targeted by the rank-one update.",
    ),
  ],
  35: [
    text(
      "editLayers",
      "Editing layers",
      "Layer indices used for multi-edit updates.",
    ),
    num("editBatch", "Edits per batch", 8, 1, 10000),
  ],
  36: [
    text(
      "editorCheckpoint",
      "Learned editor checkpoint",
      "A trained editor compatible with the base model.",
    ),
  ],
  37: [
    text(
      "memory",
      "Correction memory reference",
      "External memory artifact; base weights are distinct.",
    ),
    text(
      "counterfactualModel",
      "Counterfactual model reference",
      "Compatible replacement component required by the recipe.",
    ),
  ],
  38: [
    text(
      "projection",
      "Preservation projection reference",
      "Projection/statistics artifact for the architecture-specific recipe.",
    ),
  ],
  39: [
    text(
      "neurons",
      "Neuron identifiers",
      "Layer and neuron indices selected by an attribution study.",
    ),
    choice("action", "Neuron intervention", ["Suppress", "Amplify", "Inspect"]),
  ],
  40: [
    text(
      "codebook",
      "Adaptive codebook reference",
      "External codebook used by the editing recipe.",
    ),
    num("radius", "Initial edit radius", 1, 0.001, 100, 0.1),
  ],
  41: [
    text(
      "cacheFilter",
      "Activation cache filter",
      "Names of activations to retain.",
    ),
    text(
      "heads",
      "Attention head selection",
      "Layer/head pairs for inspection.",
    ),
  ],
  42: [
    choice("intervention", "Intervention type", [
      "Trace only",
      "Replace activation",
      "Scale activation",
    ]),
    num("scale", "Activation scale", 1, 0, 10, 0.1),
  ],
  43: [
    text(
      "representation",
      "Representation configuration",
      "Model-specific intervention configuration.",
    ),
    text(
      "positions",
      "Token positions",
      "Source and destination token indices.",
    ),
  ],
  44: [
    num("rank", "Representation rank", 4, 1, 1024),
    text(
      "positions",
      "Intervention positions",
      "Positions to train or apply the representation intervention.",
    ),
    ...training(),
  ],
  45: [
    text(
      "traceModules",
      "Trace modules",
      "Module names whose intermediate values should be captured.",
    ),
    text(
      "editFunction",
      "Edit function reference",
      "Custom intervention function stored separately; not executed here.",
    ),
  ],
  46: [
    text(
      "sae",
      "SAE checkpoint reference",
      "A sparse autoencoder compatible with the chosen activation space.",
    ),
    text("features", "Feature indices", "Sparse feature IDs to inspect."),
    choice("saeAction", "SAE operation", [
      "Inspect existing SAE",
      "Plan SAE training",
    ]),
  ],
  47: [
    text(
      "nodes",
      "Graph node selectors",
      "Operator or tensor names to modify.",
    ),
    choice("rewrite", "Graph change", [
      "Replace operator",
      "Reconnect tensors",
      "Remove node",
    ]),
    text(
      "shapeSpec",
      "Tensor shape specification",
      "Expected input/output shapes for validation.",
    ),
  ],
  48: [
    choice("level", "Optimization level", ["Basic", "Extended", "All"]),
    text(
      "provider",
      "Execution provider",
      "Provider name compatible with the installed runtime and hardware.",
    ),
  ],
  49: [
    block(
      "passes",
      "Optimization passes",
      "Ordered pass configuration for an installed Olive version.",
      "[]",
    ),
    text(
      "accelerator",
      "Target accelerator",
      "Hardware specification to validate before execution.",
    ),
  ],
  50: [
    choice("traceMode", "Tracing approach", [
      "Symbolic trace",
      "Custom tracer",
    ]),
    text(
      "rewritePass",
      "Rewrite pass reference",
      "Python rewrite function or pass reference; no execution in the prototype.",
    ),
    block(
      "exampleInputs",
      "Example input specification",
      "Tensor shapes and dtypes for tracing and validation.",
      "{}",
    ),
  ],
};
export function projectedToolFields(id: string): Field[] {
  const n = Number(id.split("-")[1]);
  if (!specifics[n]) return [];
  const common = [
    text(
      "modelReference",
      "Model / checkpoint reference",
      "Optional local source reference. Connected model inputs remain authoritative in the existing demos.",
    ),
    text(
      "outputReference",
      "Output artifact reference",
      "Proposed output location only. The browser never writes a checkpoint to this path.",
    ),
  ];
  let family: Field[] = [];
  if (n <= 10)
    family = [
      targets(),
      num("pruneRatio", "Pruning ratio", 0.1, 0, 0.95, 0.05),
      ...calibration(),
    ];
  else if (n <= 24) family = [targets(), ...calibration()];
  else if (n <= 32 && ![29, 30, 31].includes(n))
    family = [
      block(
        "sources",
        "Source checkpoints",
        "One source reference per line. Real merge compatibility has not been checked.",
      ),
      text(
        "baseModel",
        "Base model reference",
        "Optional base model used by a merge recipe.",
      ),
    ];
  else if (n >= 33 && n <= 40)
    family = [
      block(
        "edits",
        "Edit requests",
        "Proposed subject, prompt, and target records.",
        "[]",
      ),
      text(
        "localityData",
        "Locality evaluation dataset",
        "Unrelated examples used to check preservation of behavior.",
      ),
      targets(),
    ];
  else if (n >= 41 && n <= 46) family = interventions();
  else if (n >= 47)
    family = [
      text(
        "graph",
        "Graph / module reference",
        "Local ONNX artifact or PyTorch module identifier; reference only.",
      ),
    ];
  return [
    ...specifics[n],
    ...family,
    ...common,
    block(
      "extra",
      "Additional backend configuration",
      "Projected JSON/YAML or notes for the adapter. Stored verbatim; not executed or schema-validated.",
    ),
  ];
}
