import type { ExecutionEngine, RunResult, Workflow } from "./types";

export type LocalDevice = {
  id: string;
  kind: string;
  name: string;
  uuid?: string;
  is_rtx_5090?: boolean;
  total_memory_bytes?: number;
  note?: string;
  pytorch_supports_architecture?: boolean;
};

export type LocalTensor = {
  name: string;
  shape: number[];
  dtype: string;
  parameters: number;
  eligible: boolean;
  reasons: string[];
};

export type LocalRegistration = {
  id: string;
  name: string;
  resolved_path: string;
  architecture: string;
  parameter_count: number;
  directory_bytes: number;
  native_dtype: string | null;
  eligible_tensors: LocalTensor[];
  tensors: LocalTensor[];
  memory_estimate: {
    weight_bytes: number | null;
    selected_tensor_bytes?: number | null;
    estimated_resident_bytes?: number | null;
    uncertainty: string;
    assumptions: string[];
    kind?: string;
  };
};

export type RunPhase =
  | "submitting"
  | "running"
  | "cancel_pending"
  | "cancelled"
  | "failed"
  | "succeeded";

const headers = {
  "Content-Type": "application/json",
  "X-Kykeon-Request": "1",
};

const INFLIGHT = "kykeon-inflight-job";

async function errorText(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.error === "string") return body.error;
  } catch {
    /* Ignore empty or non-JSON failure pages. */
  }
  return response.statusText || "Local backend request failed.";
}

export async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) throw Error(await errorText(response));
  return response.json();
}

function remember(submissionId: string, jobId: string, cancelRequested = false) {
  try {
    sessionStorage.setItem(INFLIGHT, JSON.stringify({ submissionId, jobId, cancelRequested }));
  } catch {
    /* Session storage can be unavailable; the in-memory id still works. */
  }
}

function forget() {
  try {
    sessionStorage.removeItem(INFLIGHT);
  } catch {
    /* The editor remains usable without session storage. */
  }
}

function proxyFailure(error: unknown): boolean {
  return (
    error instanceof Error &&
    /^(Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)$/.test(
      error.message,
    )
  );
}

export function inflightJob(): { submissionId: string; jobId: string; cancelRequested?: boolean } | null {
  try {
    const raw = sessionStorage.getItem(INFLIGHT);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { submissionId?: string; jobId?: string; cancelRequested?: boolean };
    if (!parsed.submissionId || !parsed.jobId) return null;
    return { submissionId: parsed.submissionId, jobId: parsed.jobId, cancelRequested: parsed.cancelRequested === true };
  } catch {
    return null;
  }
}

type Poll = {
  events: { event?: string; nodeId?: string; state?: string; detail?: string }[];
  offset: number;
  status: string;
  error?: string | null;
};

type Hooks = { onPhase?: (phase: RunPhase) => void };
type OnEvent = Parameters<ExecutionEngine["run"]>[1];
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function resumeLocalJob(
  onEvent: OnEvent,
  signal: AbortSignal,
  hooks?: Hooks,
): Promise<RunResult> {
  const pending = inflightJob();
  if (!pending) throw Error("No local job is awaiting recovery.");
  return monitorJob(pending.jobId, onEvent, signal, hooks);
}

async function monitorJob(jobId: string, onEvent: OnEvent, signal: AbortSignal, hooks?: Hooks): Promise<RunResult> {
  let cancelWanted = signal.aborted || inflightJob()?.cancelRequested === true;
  let cancelAcknowledged = false;
  const requestCancel = () => {
    cancelWanted = true;
    const pending = inflightJob();
    if (pending) remember(pending.submissionId, jobId, true);
    hooks?.onPhase?.("cancel_pending");
  };
  signal.addEventListener("abort", requestCancel);
  if (cancelWanted) requestCancel();
  hooks?.onPhase?.(cancelWanted ? "cancel_pending" : "running");
  let offset = 0;
  let failures = 0;
  try {
    for (;;) {
      if (cancelWanted && !cancelAcknowledged) {
        try {
          const response = await fetch(`/api/v1/jobs/${jobId}/cancel`, { method: "POST", headers });
          if (response.ok) {
            const acknowledgement = await response.json();
            cancelAcknowledged = ["cancel_requested", "cancelled", "failed", "succeeded"].includes(acknowledgement.status);
          }
        } catch { /* Retry on the next poll; an HTTP abort is not cancellation. */ }
      }
      let poll: Poll;
      try {
        poll = await readJson<Poll>(await fetch(`/api/v1/jobs/${jobId}/events?offset=${offset}`));
        failures = 0;
      } catch {
        if (++failures >= 3)
          throw Error("Lost contact with the local job. Reload or press Run to reconnect; the job may still be running. Simulation was not run.");
        await delay(400);
        continue;
      }
      offset = poll.offset;
      for (const event of poll.events) {
        if (event.event === "node" && event.nodeId && event.state)
          onEvent({ nodeId: event.nodeId, state: event.state === "running" ? "running" : "complete", detail: event.detail || "" });
      }
      if (poll.status === "succeeded") {
        // Keep recovery information if the final download fails.
        const result = await readJson<{ text?: string; report?: string; model?: RunResult["model"] }>(
          await fetch(`/api/v1/jobs/${jobId}/result`),
        );
        hooks?.onPhase?.("succeeded");
        forget();
        return { text: result.text, report: result.report, model: result.model, values: result };
      }
      if (poll.status === "cancelled" || poll.status === "failed") {
        hooks?.onPhase?.(poll.status);
        forget();
        throw Error(poll.error || (poll.status === "cancelled" ? "Run cancelled." : "Local job failed."));
      }
      await delay(200);
    }
  } finally {
    signal.removeEventListener("abort", requestCancel);
  }
}

export function createLocalEngine(deviceId: string, hooks?: Hooks): ExecutionEngine {
  return {
    async run(workflow: Workflow, onEvent, signal) {
      if (inflightJob()) return resumeLocalJob(onEvent, signal, hooks);
      if (!deviceId) throw Error("Select an execution device. Kykeon does not assume GPU index 0 and will not switch to CPU.");
      if (signal.aborted) throw Error("Run cancelled.");
      hooks?.onPhase?.("submitting");
      const submissionId = crypto.randomUUID();
      try {
        const validated = await fetch("/api/v1/workflows/validate", {
          method: "POST", headers, body: JSON.stringify({ workflow, device_id: deviceId }),
        });
        if (!validated.ok) throw Error(await errorText(validated));
      } catch (error) {
        if (error instanceof TypeError || proxyFailure(error))
          throw Error("Local backend is unavailable. Simulation was not run.");
        throw error;
      }
      if (signal.aborted) throw Error("Run cancelled.");
      // Backend uses this UUID as the job ID, including when the response is lost.
      remember(submissionId, submissionId);
      let jobId = submissionId;
      try {
        const created = await fetch("/api/v1/jobs", {
          method: "POST", headers,
          body: JSON.stringify({ workflow, device_id: deviceId, submission_id: submissionId }),
        });
        if (!created.ok) {
          if (created.status >= 400 && created.status < 500) forget();
          throw Error(await errorText(created));
        }
        jobId = (await created.json()).id;
      } catch (error) {
        if (!(error instanceof TypeError) && !proxyFailure(error)) throw error;
        // A lost response can race preparation. Retry lookup, never submit a
        // second job under a new identity. Keep the ID if recovery is offline.
        let recovered = false;
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const response = await fetch(`/api/v1/submissions/${submissionId}`);
            if (response.ok) { jobId = (await response.json()).id; recovered = true; break; }
          } catch { /* Keep the identity for reload recovery. */ }
          await delay(400);
        }
        if (!recovered) throw Error("Submission status is unknown. Reload or press Run to reconnect. Simulation was not run.");
      }
      remember(submissionId, jobId);
      return monitorJob(jobId, onEvent, signal, hooks);
    },
  };
}
