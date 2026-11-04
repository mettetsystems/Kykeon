import type { LocalDevice, LocalRegistration, LocalTensor } from "./local-engine";

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unavailable";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function LocalCheckpointPanel({
  backend,
  devices,
  deviceId,
  onDevice,
  path,
  onPath,
  registrations,
  busy,
  onRegister,
  onPlace,
  onRefresh,
}: {
  backend: string;
  devices: LocalDevice[];
  deviceId: string;
  onDevice: (id: string) => void;
  path: string;
  onPath: (path: string) => void;
  registrations: LocalRegistration[];
  busy: boolean;
  onRegister: () => void;
  onPlace: (registration: LocalRegistration) => void;
  onRefresh: () => void;
}) {
  return (
    <div className="local-panel">
      <div className="section-label">
        LOCAL CHECKPOINT<span>{registrations.length}</span>
      </div>
      <p className="local-note">
        {backend === "ok"
          ? "Backend connected. Registration inspects files; it does not download a model."
          : backend === "unavailable"
            ? "Local backend unavailable. Simulation still runs and will not pretend to be local."
            : "Checking the local backend…"}
      </p>
      <label className="field compact">
        <span>Execution device</span>
        <select
          aria-label="Execution device"
          value={deviceId}
          disabled={busy}
          onChange={(event) => onDevice(event.target.value)}
        >
          <option value="">Select a device</option>
          {devices.map((device) => (
            <option key={device.id} value={device.id}>
              {device.name} · {device.id}
              {device.is_rtx_5090 ? " · RTX 5090" : ""}
            </option>
          ))}
        </select>
        <small>GPU index 0 is not assumed. CPU is used only if you select it.</small>
      </label>
      <label className="field compact">
        <span>Checkpoint directory</span>
        <input
          aria-label="Checkpoint directory"
          value={path}
          disabled={busy}
          onChange={(event) => onPath(event.target.value)}
          placeholder="/path/inside/a/model/root"
        />
      </label>
      <div className="local-actions">
        <button disabled={busy || backend !== "ok"} onClick={onRegister}>
          Register
        </button>
        <button disabled={busy || backend !== "ok"} onClick={onRefresh}>
          Refresh
        </button>
      </div>
      {registrations.map((registration) => (
        <button
          className="catalog-item"
          key={registration.id}
          disabled={busy}
          onClick={() => onPlace(registration)}
        >
          <span>
            <strong>{registration.name}</strong>
            <small>
              {registration.architecture} · real checkpoint ·{" "}
              {registration.eligible_tensors.length} projections
            </small>
          </span>
        </button>
      ))}
    </div>
  );
}

export function TensorDetails({ tensor }: { tensor: LocalTensor | undefined }) {
  if (!tensor) return null;
  const bytes =
    tensor.dtype === "F32"
      ? tensor.parameters * 4
      : tensor.dtype === "F16" || tensor.dtype === "BF16"
        ? tensor.parameters * 2
        : null;
  return (
    <div className="help-card">
      <strong>{tensor.name}</strong>
      <p>
        Shape {tensor.shape.join(" × ")} · {tensor.dtype} · {tensor.parameters.toLocaleString()}{" "}
        parameters · about {formatBytes(bytes)} for this tensor
      </p>
      <small>
        Working-memory figures are preflight estimates with high uncertainty. They are not a
        guarantee against running out of memory.
      </small>
    </div>
  );
}
