import { describe, expect, it, vi } from "vitest";
import { createLocalEngine } from "./local-engine";
import { readStorage, writeStorage } from "./storage";
import { starter } from "./engine";

function workflow() {
  return starter("inference");
}

describe("local execution cancellation and recovery", () => {
  it("waits for a terminal cancellation instead of treating an aborted request as completion", async () => {
    const phases: string[] = [];
    const calls: string[] = [];
    let polls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push(`${init?.method || "GET"} ${url}`);
        if (String(url).includes("/validate"))
          return { ok: true, json: async () => ({ accepted: true }) };
        if (String(url).endsWith("/jobs") && init?.method === "POST")
          return { ok: true, json: async () => ({ id: "job-1" }) };
        if (String(url).includes("/cancel"))
          return { ok: true, json: async () => ({ status: "cancel_requested" }) };
        polls += 1;
        return {
          ok: true,
          json: async () =>
            polls < 2
              ? { events: [], offset: 0, status: "running", error: null }
              : { events: [], offset: 0, status: "cancelled", error: "Run cancelled." },
        };
      }),
    );
    const controller = new AbortController();
    const engine = createLocalEngine("cpu", { onPhase: (phase) => phases.push(phase) });
    const promise = engine.run(workflow(), () => {}, controller.signal);
    controller.abort();
    await expect(promise).rejects.toThrow(/cancelled/);
    expect(calls.some((call) => call.includes("/cancel"))).toBe(true);
    expect(phases).toContain("cancel_pending");
    expect(phases.indexOf("cancel_pending")).toBeLessThan(phases.indexOf("cancelled"));
    vi.unstubAllGlobals();
  });

  it("recovers a job when the submission response is lost", async () => {
    let posts = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (String(url).includes("/validate"))
          return { ok: true, json: async () => ({}) };
        if (String(url).endsWith("/jobs") && init?.method === "POST") {
          posts += 1;
          throw new TypeError("network down");
        }
        if (String(url).includes("/submissions/"))
          return { ok: true, json: async () => ({ id: "job-recovered", status: "running" }) };
        if (String(url).includes("/events"))
          return {
            ok: true,
            json: async () => ({ events: [], offset: 4, status: "succeeded", error: null }),
          };
        if (String(url).includes("/result"))
          return { ok: true, json: async () => ({ text: "real", execution: "local" }) };
        throw Error(String(url));
      }),
    );
    const result = await createLocalEngine("cpu").run(
      workflow(),
      () => {},
      new AbortController().signal,
    );
    expect(posts).toBe(1);
    expect(result.text).toBe("real");
    expect(result.values.execution).toBe("local");
    vi.unstubAllGlobals();
  });

  it("retries polling and then reports a confirmed cancellation", async () => {
    let polls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (String(url).includes("/validate"))
          return { ok: true, json: async () => ({}) };
        if (init?.method === "POST")
          return { ok: true, json: async () => ({ id: "job-2" }) };
        polls += 1;
        if (polls < 3) throw new TypeError("temporary");
        return {
          ok: true,
          json: async () => ({
            events: [],
            offset: 0,
            status: "cancelled",
            error: "Run cancelled.",
          }),
        };
      }),
    );
    await expect(
      createLocalEngine("cpu").run(workflow(), () => {}, new AbortController().signal),
    ).rejects.toThrow(/cancelled/);
    expect(polls).toBe(3);
    vi.unstubAllGlobals();
  });
});

describe("backend disconnection", () => {
  it("does not run a simulation when the dev proxy reports the API is down", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => {
          throw Error("not json");
        },
      })),
    );
    await expect(
      createLocalEngine("cpu").run(workflow(), () => {}, new AbortController().signal),
    ).rejects.toThrow("Local backend is unavailable. Simulation was not run.");
    vi.unstubAllGlobals();
  });
});

describe("browser storage failures", () => {
  it("reports a storage write failure without throwing", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw Error("blocked");
      },
      setItem: () => {
        throw Error("quota");
      },
    });
    expect(readStorage("kykeon-workspace-v1")).toBeNull();
    expect(writeStorage("kykeon-workspace-v1", "{}")).toMatch(/quota/);
    vi.unstubAllGlobals();
  });
});
