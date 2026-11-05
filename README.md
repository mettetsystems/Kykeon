# Kykeon

A local, visual workbench for inference controls and bounded model surgery. The browser keeps an explicit **Simulation** mode. **Local execution** is a separate loopback backend and never falls back to simulation.

## Quick start with Make (Linux)

Requires Python 3.14, Node.js 22+, npm, and GNU Make. The install target uses
this project's pinned CUDA backend dependencies; it does not install drivers,
download checkpoints, or require root privileges.

```bash
make setup    # Create .venv and local directories; safe to repeat
make install  # Install npm lockfile and pinned Python dependencies
make start    # Start frontend + backend in the background
make stop     # Gracefully stop only services launched by these scripts
```

`make install` also runs setup, so on a fresh checkout you can start with
`make install && make start`. You can also run `./setup.sh`, `./install.sh`,
`./start.sh`, and `./stop.sh` directly from any directory. Set `KYKEON_PYTHON`
if your Python 3.14 executable has a different name.

Open **http://127.0.0.1:5173/**. Backend documentation is at
**http://127.0.0.1:8765/docs**. Logs live in `.kykeon/logs/` and process records
in `.kykeon/run/`. Repeating start/stop is safe. Occupied ports are reported
instead of killing another server or silently switching ports.

The scripts default to `.kykeon/models/` for registered checkpoints and
`.kykeon/artifacts/` for derived files. Export `KYKEON_MODEL_ROOTS` (colon-separated)
and/or `KYKEON_ARTIFACT_ROOT` before `make start` to override these directories.
Model files are never removed by `make stop`. Both services bind to loopback;
ports 5173 and 8765 match the existing frontend proxy. These commands run the
local development app, not a production deployment.

## Run the workbench

Requires Node.js 22+ and npm.

```bash
npm install
npm run dev
```

Open the localhost URL printed by Vite. Without the backend, Simulation still runs. Local execution reports that the backend is unavailable.

```bash
npm run typecheck
npm test
npm run build
```

## Run local execution

The backend uses its own virtual environment. System Python on this machine is 3.14, which has a matching PyTorch wheel. Do not reuse an unrelated CUDA or PyTorch install.

PyTorch 2.15.0 from the CUDA 13.2 index is the current stable build that includes `sm_120` (RTX 5090). Official guidance: Blackwell needs a CUDA 12.8+ wheel, and current stable wheels are built with CUDA 12.8 or newer. This driver reports CUDA 13.4, so the CUDA 13.2 wheel is the matching stable choice. The CPU-only PyPI torch wheel is not a substitute.

```bash
python3.14 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.venv/bin/pip install -r backend/requirements.txt --upgrade-strategy only-if-needed
```

Point the backend at directories you control. It rejects paths outside those roots, including symlink escapes.

```bash
export KYKEON_MODEL_ROOTS="$PWD/.kykeon/models"
export KYKEON_ARTIFACT_ROOT="$PWD/.kykeon/artifacts"
PYTHONPATH=backend .venv/bin/python -m kykeon_backend.tiny_fixture --dest .kykeon/models/tiny-llama
PYTHONPATH=backend .venv/bin/python -m kykeon_backend
```

The API listens on `127.0.0.1:8765`. Vite proxies `/api` to it. Each page session starts in Simulation. Importing or loading a saved workflow returns to Simulation until you explicitly choose Local. A remembered device is matched by GPU UUID, not by CUDA index. Checkpoint files, safetensors index entries, and tokenizer path references must stay inside the configured roots. Local execution follows one connected chain and rejects a dangling branch. CPU jobs do not initialize CUDA.

```bash
npm run test:backend
```

The tiny fixture is generated locally for tests and the browser path. It is not a 3B model. No large checkpoint is downloaded.

## Explore

The inference starter is loaded on first launch. Select any node on the canvas to see detailed sidebar documentation: what it does, when to use it, where it fits in a workflow, and expandable considerations and limitations. Its editable settings appear below the description. All 150 catalog entries and the workflow utility nodes have individual practical examples. Add nodes from the searchable library, connect compatible typed ports, and run a simulation. Use the starter menu for checkpoint merging or quantization. Model Surgery and Inference tabs load their respective starter workflows; save a named workflow before switching if you want to return to it.

- Drag nodes to arrange; scroll to zoom; use the minimap and fit-view controls to navigate.
- Undo/redo with toolbar buttons or Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z.
- Select a node to duplicate or delete it. Delete/Backspace also removes the selected node outside text fields.
- Double-click an edge to remove it.
- Save stores a named snapshot in this browser. Export downloads versioned JSON; Import validates a workflow before replacing the current canvas.
- A successful surgery simulation prepares a custom model. **Export custom model** downloads a metadata manifest and adds the simulated model to **Your models**, where it can be reused in inference.

The complete reference catalog has 50 model types across 10 categories, 50 inference techniques, and 50 surgery tools. Supported simulation nodes are labeled; other entries are catalog-only and block runs until removed. The catalog is a taxonomy, not a popularity ranking, and categories may overlap conceptually.

## Simulation behavior

Simulation is selected explicitly and stays separate from local execution. Fixtures explain text generation, merging, or quantization according to the prompt. Other prompts receive a generic illustrative response. Temperature, top-K, top-P, seed, repetition penalty, output length, and stop strings affect deterministic fixture selection or truncation. The simulation counts whitespace-delimited words as tokens. These fixtures illustrate controls; they do not reproduce real sampling distributions or predict model quality.

Merging checks architecture identifiers, records source contributions, and preserves transformation history. Quantization uses simple precision-based size scaling. Baseline sizes, quality descriptions, and all outputs are illustrative, not measurements. No GPU access is required for simulation. In simulation, local model paths are strings and are not opened. A simulated custom-model export contains metadata, not weights. Local execution, when selected, opens only a registered directory and writes derived checkpoints under the artifact root.

Workflows and custom models persist in local storage without a server or account. Storage errors are reported; JSON downloads remain available. Clearing browser storage clears saved work. No cloud publication or analytics is configured.

## Structure

- `src/catalog.json` retains names, descriptions, and category assignments extracted from the supplied product brief.
- `src/catalog.ts` defines supported nodes, typed ports, and configuration fields.
- `src/types.ts` defines workflow, model lineage, manifest, and execution contracts.
- `src/engine.ts` implements graph validation, import validation, templates, and a cancellable `ExecutionEngine` simulation adapter.
- `src/main.tsx` implements the editor, persistence, inspector, and result views.

A future execution backend can implement `ExecutionEngine` while retaining the workflow UI. The local backend in `backend/` is that adapter for dense Llama inference and operation-4 weight rescaling. Other catalog entries remain references.

## Inference Optimization and Performance Validation

Two additional top-level workspaces provide simulated performance pipelines:

- **Inference Optimization:** model → runtime configuration → workload → benchmark → comparison. Adjust batching, KV-cache status, context budget, request count, input/output lengths, and concurrency. Compare against the same model and workload with batch size one and caching disabled.
- **Performance Validation:** model → runtime configuration → workload → benchmark → acceptance gate. Set maximum p95 request latency, minimum aggregate throughput, and maximum memory. The report lists each criterion and an overall PASS/FAIL.

All figures are deterministic educational formulas, not hardware measurements. The synthetic p95 value is not calculated from measured request samples. These pipelines neither execute a serving backend nor assess answer quality. A simulated PASS is not deployment approval. Both workflows support saving, JSON import/export, cancellation, and selected-node documentation.

## Enterprise and local resource scope

The **Operations** library contains the supplied 50-level surgery taxonomy, separately from the 50 surgery tools. Flags appear in the library, canvas, and selected-node sidebar. Search for `Enterprise` to find flagged operations.

The planning baseline is one RTX 5090, an approximately 3B pretrained model, and adaptation no heavier than PEFT/LoRA. **Local candidate** means a bounded compatible recipe is a candidate, not a measured guarantee. **Recipe-dependent** means the operation needs an architecture, transformation extent, memory plan, and recovery-training budget before classification. **Enterprise** means the catalog recipe exceeds the intended local scope; it is not a paywall or proof that no variant can run locally.

Levels 48–50 are provisionally Enterprise for broad representation reconstruction, full-student architecture distillation, and predominantly from-scratch reimplementation. Small or adapter-only distillation needs its own assessment. Structural conversions remain recipe-dependent rather than being classified by level number alone. No recipe has been benchmarked on the user's GPU. These labels grant no execution capability; all 50 operation references remain catalog-only. Generic tools are not labeled Enterprise based on their most expensive possible use.

Background: [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) reduces trainable parameter requirements; [MambaInLlama's 3B recipe](https://github.com/jxiw/MambaInLlama/blob/main/llama3.2_3B/README.md) illustrates why architecture-conversion requirements must be evaluated at the recipe level. Neither establishes a universal impossibility threshold for a single 5090.

The Optimize library includes three standalone runtime-tuning nodes: **KV cache optimization** (enabled), **Batch size optimization** (4), and **Context budget optimization** (2,048 input tokens). The optimization starter connects all three between Runtime configuration and Benchmark workload. Nodes change only their own setting; downstream nodes override earlier values for the same setting. Existing saved workflows remain unchanged—load “Optimize inference serving” to use the expanded starter. Defaults are educational fixture settings, not benchmarked 5090 recommendations.

## Custom source nodes

Every workspace has an **Add custom node** button. Enter a label, choose JSON, PyTorch, or TensorFlow, select input/output port types, and paste source or import a `.json`, `.py`, or `.txt` file. JSON must be a valid object; labels are limited to 80 characters and source to 100,000 characters (file uploads are limited to 100 KB). Changing format preserves existing source; the explicit example button replaces it.

Custom source is stored in the node, persists with the workflow, and survives JSON export/import and duplication. Select the node and choose **Edit source** to revise it. Changing port types disconnects its existing edges so they can be reconnected consistently. All custom nodes are **catalog-only**: neither Python nor custom JSON executes in this browser. Python syntax and dependencies are not checked; a future backend needs an explicit execution contract. Source is kept locally, with no upload to an external service.

## Projected tool configuration

All 50 entries in **Tools** have editable projected fields, covering their intended workflow: pruning, compression, merging, adapters, knowledge editing, interventions, or graph transformation. Select a tool node to edit its complete field set in the inspector; multiline fields accept recipe text, source lists, and edit requests. Planned values are retained by save, duplication, and workflow JSON export/import. Legacy workflows receive only missing projected defaults during import/restore; existing values are preserved.

**Planned** labels distinguish these fields from the existing active simulation controls. They are proposed Kykeon interface fields—not a verified mapping to a particular backend release. Defaults are illustrative, local paths are references only, and advanced text is stored verbatim. No backend method, dataset, target module, or tensor compatibility is validated. Changing the planned merge method to SLERP does not change the existing linear-merge fixture; similarly, projected TorchAO options do not alter its precision-only fixture.

Reference anchors used for the projected interfaces: [MergeKit configuration](https://github.com/arcee-ai/mergekit), [PEFT LoRA configuration](https://huggingface.co/docs/peft/main/package_reference/lora), [TorchAO INT4 configuration](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int4WeightOnlyConfig.html), and [EasyEdit](https://github.com/zjunlp/EasyEdit). Other fields describe intended Kykeon workflows rather than asserting exact upstream API coverage.

## Execution recovery and publication

Job identity is retained in session storage before submission. Reloading the page
resumes monitoring without submitting another job or enabling Local mode. After a
connection loss, **Run** reconnects to the pending job. Cancellation requests are
retried until acknowledged, and recovery information remains until the result is
retrieved or a failed/cancelled terminal state is confirmed.

The backend writes a result first, then atomically publishes a `success.json`
record with result and provenance hashes. Only validated success records expose
artifacts. Cancellation wins until that record is committed; after commitment the
job is successful. Restart restores job lookup records and marks interrupted jobs
failed. Older exports without a success record are retained on disk but hidden
from the verified artifact list. Artifact directories must be user-owned; output
symlinks below the configured root are rejected.

Worker output is captured in each job's `worker.log`; timeout failures include its
last output. CPU selection does not probe CUDA. A sandbox that blocks local
socket/thread wakeups can hang Starlette's TestClient even for an empty FastAPI
app. Run backend integration tests in a normal local shell in that environment;
this is distinct from a model/GPU execution stall.
