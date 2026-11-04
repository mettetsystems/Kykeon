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

function remember(submissionId: string, jobId: string) {
  try {
    sessionStorage.setItem(INFLIGHT, JSON.stringify({ submissionId, jobId }));
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

export function inflightJob(): { submissionId: string; jobId: string } | null {
  try {
    const raw = sessionStorage.getItem(INFLIGHT);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { submissionId?: string; jobId?: string };
    if (!parsed.submissionId || !parsed.jobId) return null;
    return { submissionId: parsed.submissionId, jobId: parsed.jobId };
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

export function createLocalEngine(
  deviceId: string,
  hooks?: { onPhase?: (phase: RunPhase) => void },
): ExecutionEngine {
  return {
    async run(workflow: Workflow, onEvent, signal) {
      if (!deviceId)
        throw Error(
          "Select an execution device. Kykeon does not assume GPU index 0 and will not switch to CPU.",
        );
      const submissionId = crypto.randomUUID();
      hooks?.onPhase?.("submitting");
      let jobId = "";
      const payload = JSON.stringify({
        workflow,
        device_id: deviceId,
        submission_id: submissionId,
      });
      try {
        const validated = await fetch("/api/v1/workflows/validate", {
          method: "POST",
          headers,
          body: JSON.stringify({ workflow, device_id: deviceId }),
        });
        if (!validated.ok) throw Error(await errorText(validated));
        const created = await fetch("/api/v1/jobs", {
          method: "POST",
          headers,
          body: payload,
        });
        if (!created.ok) throw Error(await errorText(created));
        jobId = (await created.json()).id;
      } catch (error) {
        if (!(error instanceof TypeError) && !proxyFailure(error)) throw error;
        try {
          const recovered = await fetch(`/api/v1/submissions/${submissionId}`);
          if (!recovered.ok)
            throw Error("Local backend is unavailable. Simulation was not run.");
          jobId = (await recovered.json()).id;
        } catch (recovery) {
          if (
            recovery instanceof Error &&
            recovery.message.includes("Simulation was not run")
          )
            throw recovery;
          throw Error("Local backend is unavailable. Simulation was not run.");
        }
      }
      remember(submissionId, jobId);
      hooks?.onPhase?.("running");
      let cancelSent = false;
      const requestCancel = () => {
        if (cancelSent) return;
        cancelSent = true;
        hooks?.onPhase?.("cancel_pending");
        fetch(`/api/v1/jobs/${jobId}/cancel`, { method: "POST", headers }).catch(
          () => undefined,
        );
      };
      if (signal.aborted) requestCancel();
      else signal.addEventListener("abort", requestCancel, { once: true });
      let offset = 0;
      let failures = 0;
      for (;;) {
        let poll: Poll;
        try {
          const response = await fetch(
            `/api/v1/jobs/${jobId}/events?offset=${offset}`,
          );
          if (!response.ok) throw Error(await errorText(response));
          poll = await response.json();
          failures = 0;
        } catch (error) {
          if (error instanceof Error && error.message.startsWith("Progress offset"))
            throw error;
          failures += 1;
          if (failures >= 3)
            throw Error(
              "Lost contact with the local job. Simulation was not run.",
            );
          await new Promise((resolve) => setTimeout(resolve, 400));
          continue;
        }
        offset = poll.offset;
        for (const event of poll.events) {
          if (event.event === "node" && event.nodeId && event.state)
            onEvent({
              nodeId: event.nodeId,
              state: event.state === "running" ? "running" : "complete",
              detail: event.detail || "",
            });
        }
        if (poll.status === "succeeded") {
          hooks?.onPhase?.("succeeded");
          forget();
          break;
        }
        if (poll.status === "cancelled") {
          hooks?.onPhase?.("cancelled");
          forget();
          throw Error(poll.error || "Run cancelled.");
        }
        if (poll.status === "failed") {
          hooks?.onPhase?.("failed");
          forget();
          throw Error(poll.error || "Local job failed. Simulation was not run.");
        }
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
      const result = await readJson<{
        text?: string;
        report?: string;
        model?: RunResult["model"];
      }>(await fetch(`/api/v1/jobs/${jobId}/result`));
      return {
        text: result.text,
        report: result.report,
        model: result.model,
        values: result,
      };
    },
  };
}
