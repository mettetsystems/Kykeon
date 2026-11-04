import { performanceHelp } from "./performance";
import examples from "./node-examples.json";
import type { Definition } from "./types";
export type NodeHelp = {
  explanation: string;
  usage: string;
  considerations: string;
  workflow: string;
};
const mechanisms: Record<string, string> = {
  "Autoregressive and language generation models":
    "This family generates a sequence from earlier context. The model supplies predictions; inference controls determine how a prediction becomes the next output element.",
  "Diffusion models":
    "This family builds samples through denoising or related refinement procedures. Its model and sampling process must be compatible; text-generation controls do not automatically apply.",
  "Generative adversarial networks (GANs)":
    "A GAN workflow involves a generator and a discriminator during training. Its controls and conditioning depend on the particular trained architecture.",
  "Variational autoencoders (VAEs)":
    "These architectures use a learned latent representation between input data and generated samples. The choice of latent representation affects which transformations and sampling methods make sense.",
  "Flow-based generative models":
    "These approaches connect a simple distribution to a data distribution through learned transformations. Invertible normalizing flows and flow-matching methods are related ideas, not interchangeable implementations.",
  "Energy-based and probabilistic models":
    "These families describe probabilities or energy values over possible configurations. Producing samples requires a sampling procedure suited to the particular model.",
  "Generative flow networks":
    "A GFlowNet constructs structured objects through a sequence of choices, aiming to produce diverse objects according to a reward-related distribution.",
  "Hybrid generative models":
    "A hybrid combines components or objectives from different model families. Inspect each component separately to understand which controls or surgery operations apply.",
  "Structured data generation":
    "These are broad capability categories for generating objects with explicit structure. The underlying implementation may use several different neural architectures.",
  "Multimodal generative models":
    "Multimodal describes the input or output capabilities of a system rather than a single architecture. A pipeline can contain different components for text, images, audio, or video.",
  "Sampling and randomness controls":
    "This control acts during next-token selection. It changes how a runtime uses the model’s predictions without modifying the model’s learned weights.",
  "Repetition, diversity, and token probability controls":
    "This technique changes candidate scores or eligibility during decoding. Its effect depends on the tokenizer, previously generated tokens, and the decoding strategy.",
  "Generation length and stopping controls":
    "This control affects the amount of generated output or the conditions under which generation ends. A stopped or truncated response still needs validation for the task.",
  "Decoding and candidate-selection strategies":
    "This technique changes how candidate completions are explored or selected. Some strategies need multiple generations or an additional scoring component, not just a sampling parameter.",
  "Structured output and constrained decoding":
    "This technique restricts the shape of allowed output during generation. Structural validity does not establish that the content is factually correct or suitable for its intended action.",
  "Prompt, context, and behavioral controls":
    "This technique changes the information or instructions around generation. It operates through prompting or orchestration rather than rewriting model weights.",
  "Model pruning and structural reduction":
    "Pruning removes or disables weights, layers, neurons, or other model structures. The exact intervention and its hardware benefits depend on the method and model.",
  "Model quantization and numerical compression":
    "Quantization represents weights or activations with reduced numerical precision. A real workflow must connect the quantization method, saved representation, and inference runtime.",
  "Model merging and weight manipulation":
    "These tools combine compatible model parameters, layers, or adapters. A usable merge depends on matching components and requires evaluation beyond successfully producing an artifact.",
  "Model knowledge editing":
    "Knowledge editing targets particular associations or behaviors while attempting to limit unrelated changes. Some methods add external components instead of permanently changing base weights.",
  "Activation intervention and mechanistic interpretability":
    "These tools inspect or change intermediate model representations. An intervention during execution is not necessarily a permanent checkpoint edit.",
  "Computational graph and architecture surgery":
    "These tools work with computational graphs, operators, and execution structure. Their purpose differs from changing a specific factual association in a language model.",
};
const utilityHelp: Record<string, NodeHelp> = {
  prompt: {
    explanation:
      "Combines a model reference with the text you want it to respond to. It starts a context stream that sampling controls can modify before generation.",
    usage:
      "Ask for an explanation of language models, merging, or quantization, then compare how different controls affect the fixture output.",
    considerations:
      "The prompt chooses an educational fixture. Arbitrary prompts do not invoke a real model and cannot produce general-purpose answers.",
    workflow: "Model → Prompt → Sampling controls → Generate text.",
  },
  sampling: {
    explanation:
      "Groups seven non-invasive controls in one node: temperature, top-K, top-P, random seed, repetition penalty, maximum output tokens, and stop sequences. It passes the updated context to generation.",
    usage:
      "Keep the prompt and seed fixed, change one setting, and inspect the comparison against default fixture settings.",
    considerations:
      "These controls drive a simplified fixture algorithm, not a real probability distribution. A downstream control overrides the same setting from an earlier control node.",
    workflow:
      "Connect a context input from Prompt or another control, then pass context to Generate text.",
  },
  generate: {
    explanation:
      "Consumes the prompt, model reference, and accumulated controls to produce a deterministic sample response. It also retains a default-settings fixture for comparison.",
    usage:
      "Run the inference starter after editing sampling settings to see a completion and visible execution progress.",
    considerations:
      "No language model executes. Output length counts whitespace-delimited words, not tokenizer tokens; the same settings and prompt reproduce the same fixture.",
    workflow: "Context → Generate text → Output comparison.",
  },
  compare: {
    explanation:
      "Displays the default-settings fixture and the configured fixture together so their differences can be inspected.",
    usage:
      "Change one inference control at a time and compare the response wording, repetition, stopping point, or length.",
    considerations:
      "The baseline is a fixture generated with default controls, not a saved previous run or a measured quality benchmark.",
    workflow:
      "Connect the generation output from Generate text. Read both outputs in the Run output panel.",
  },
  evaluate: {
    explanation:
      "Summarizes a simulated model transformation, its history, and an illustrative size comparison against the 2 GB sample baseline. The model reference passes through unchanged.",
    usage:
      "Place this after a merge or quantization node to inspect the transformation before exporting its manifest.",
    considerations:
      "No accuracy, latency, VRAM, or capability tests are actually performed. Real deployment requires representative measurements and acceptance criteria.",
    workflow: "Transformed model → Evaluate model → Custom model export.",
  },
  export: {
    explanation:
      "Makes the transformed model available for export after the run completes. The export action downloads a JSON manifest and registers a reusable custom-model entry in this browser.",
    usage:
      "Run a surgery starter, click Export custom model in the results, then add the registered entry to an inference workflow.",
    considerations:
      "The manifest contains source paths, architecture, illustrative size, and transformation history. It contains no model weights and does not publish a checkpoint to a directory.",
    workflow:
      "Connect a transformed or evaluated model. After execution, use the explicit export button in Run output.",
  },
  custom: {
    explanation:
      "Reuses a model record created by a previous surgery simulation. Its source references and transformation history travel with the node.",
    usage:
      "Replace the sample model node in an inference workflow with this custom model and reconnect it to Prompt.",
    considerations:
      "This is a metadata record, not a loaded model. Reusing it does not make generated text reflect real merged or quantized weights.",
    workflow: "Custom model → Prompt → Sampling controls → Generate text.",
  },
};
export function nodeHelp(d: Definition): NodeHelp {
  if (d.id === "custom-code")
    return {
      explanation:
        "Stores a labeled custom operation as JSON configuration, PyTorch source, or TensorFlow source. The input and output types describe where it belongs in your graph.",
      usage:
        "Use this node to bring your own transformation or validation code into a workflow design, then export the workflow to preserve the complete source.",
      considerations:
        "Source is stored as text and is never evaluated in this browser. JSON syntax is checked; Python syntax, dependencies, tensor compatibility, and runtime behavior are not verified. This node blocks simulated runs until removed.",
      workflow:
        "Connect compatible typed ports. Use Edit source in the inspector to change its label, format, code, or port types. A future backend is required for execution.",
    };
  if (d.id === "operation-4")
    return {
      explanation:
        "Multiplies one selected attention or MLP projection by a finite scale factor and writes a new checkpoint. Level 4 is a parameter-level change, not a measured training cost. Local execution can apply it to a dense Llama checkpoint. Simulation does not read or modify weights.",
      usage:
        "Register a local Llama checkpoint, choose one exact projection such as a q_proj or up_proj weight, keep the scale between 0.5 and 1.5, and run Local execution so Kykeon can export and reload the derived checkpoint.",
      considerations: `${d.resourceProfile?.reason} The first local adapter rejects embeddings, output heads, biases, quantized tensors, and tied parameters. A scale of 1.0 preserves values. A saved checkpoint is a record of the edit, not evidence that the model improved. Seeded generations are not bitwise-stable across devices or library versions.`,
      workflow:
        "Registered checkpoint → Weight rescaling → Evaluate model → Custom model export. This path runs only in Local execution.",
    };
  if (d.resourceProfile)
    return {
      explanation: `${d.description} This is level ${d.resourceProfile.level} in the proposed ${d.category} taxonomy. The level describes the scope of modification, not a measured cost or guaranteed difficulty.`,
      usage: `Use ${d.name.toLowerCase()} as a planning step when evaluating how to derive a new model from an existing checkpoint. Select a compatible architecture, define the exact affected components, and establish retention criteria before implementation.`,
      considerations: d.resourceProfile.reason,
      workflow:
        "Catalog-only operation reference. Plan transformation, optional PEFT recovery, and validation as separate stages. This node cannot execute or export real model weights.",
    };
  if (performanceHelp[d.id]) return performanceHelp[d.id];
  if (utilityHelp[d.id]) return utilityHelp[d.id];
  const usage = examples[d.id as keyof typeof examples];
  if (!usage) throw Error(`Missing node documentation: ${d.id}`);
  let considerations =
    d.kind === "model"
      ? "Choose a checkpoint whose actual architecture and modality match this entry. A taxonomy label alone does not guarantee support for a runtime or operation."
      : d.kind === "control"
        ? "Support and parameter meanings vary between runtimes. Apply compatible controls and compare one change at a time; more controls do not automatically produce better output."
        : "Retain the baseline, apply one well-defined transformation, and evaluate capability retention, the target improvement, resource use, and unrelated behavior before promoting an artifact.";
  if (d.category === "Model pruning and structural reduction")
    considerations +=
      " Sparse weights do not automatically give a speedup without suitable kernels.";
  if (
    d.category === "Model knowledge editing" ||
    d.category === "Activation intervention and mechanistic interpretability"
  )
    considerations +=
      " Record any extra memory or intervention components needed to reproduce the behavior.";
  if (d.id === "surgery-25")
    considerations =
      "The simulation accepts matching architecture identifiers and uses A + B weights summing to one. It records lineage but does not combine tensors; real compatibility requires more than a matching name.";
  if (d.id === "surgery-11")
    considerations =
      "The simulation scales the sample size by precision: FP16 ×1, INT8 ×0.5, INT4 ×0.25. Actual memory, speed, and quality must be measured with a compatible quantization backend.";
  return {
    explanation: `${d.description}. ${mechanisms[d.category] || ""}`,
    usage,
    considerations,
    workflow: d.supported
      ? d.kind === "model"
        ? "Use as a model source before Prompt, or connect it to a supported surgery node. Paths are references only."
        : d.kind === "control"
          ? "Place after Prompt or another context control and before Generate text. Later settings override earlier settings with the same name."
          : "Connect compatible sample models, then Evaluate model and Custom model export."
      : "Catalog reference only. You may place this node to sketch a future workflow, but it blocks simulated execution. Displayed ports are placeholders, not a supported integration contract.",
  };
}
