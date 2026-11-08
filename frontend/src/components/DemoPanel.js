import { AlertTriangle, Camera, CircleDot, Hand, Radio, Wifi, WifiOff, X } from "lucide-react";
import { DIMENSION_LABELS, STATES, STATE_NAMES, TONE } from "../config";

function Bar({ label, value, tone = "brand" }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div>
      <div className="mb-0.5 flex justify-between text-[11px] text-zinc-400">
        <span>{label}</span>
        <span className="tabular-nums">{value == null ? "–" : `${pct}%`}</span>
      </div>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-shell-700">
        <div className={`h-full rounded-full transition-all duration-500 ${pct > 60 ? "bg-rose-400" : pct > 40 ? "bg-amber-400" : "bg-emerald-400"}`} style={{ width: `${pct}%` }} />
        <span className="absolute inset-y-0 left-[60%] w-px bg-zinc-400/60" title="High threshold (60%)" />
      </div>
    </div>
  );
}

/**
 * The demo control panel (Plan section D):
 *  - Manual: override buttons for each state (Option B, the reliable backup)
 *  - Auto:   live webcam + mic sensing through the trained model (Option A)
 */
export default function DemoPanel({
  open, onClose, controlMode, onControlMode, cognitiveState, onManualState,
  backend, live, liveEnabled, onToggleLive, lastPrediction,
}) {
  if (!open) return null;
  const { online, wsConnected, health } = backend;
  const probs = lastPrediction?.smoothed && Object.keys(lastPrediction.smoothed).length ? lastPrediction.smoothed : lastPrediction?.probabilities;

  return (
    <section
      aria-label="Adaptation control panel"
      className="absolute bottom-4 left-4 z-30 w-[300px] animate-toast-in rounded-xl border border-shell-700 bg-shell-900/95 p-4 text-sm shadow-2xl shadow-black/50 backdrop-blur"
    >
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-semibold">Adaptation controls</h2>
        <button onClick={onClose} aria-label="Close controls" className="rounded p-1 text-zinc-400 hover:bg-shell-700 hover:text-white">
          <X size={14} />
        </button>
      </div>

      {/* Backend status */}
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md bg-shell-800 px-3 py-2 text-[11px] text-zinc-300">
        <span className="flex items-center gap-1.5">
          {online ? <Wifi size={12} className="text-emerald-400" /> : <WifiOff size={12} className="text-rose-400" />}
          Backend {online ? "online" : "offline"}
        </span>
        {online && (
          <span className={health?.mode === "model" ? "text-emerald-300" : "text-amber-300"}>
            {health?.mode === "model" ? `Model: ${health.dimensions.length} loaded` : "Mock model (no checkpoint yet)"}
          </span>
        )}
        {online && <span className="text-zinc-500">WS {wsConnected ? "live" : "reconnecting"}</span>}
      </div>

      {/* Mode switch */}
      <div className="mb-3 grid grid-cols-2 rounded-md bg-shell-800 p-0.5 text-xs" role="radiogroup" aria-label="Control mode">
        {[
          ["manual", "Manual", Hand],
          ["auto", "Auto (model)", Radio],
        ].map(([id, label, Icon]) => (
          <button
            key={id}
            role="radio"
            aria-checked={controlMode === id}
            onClick={() => onControlMode(id)}
            className={`flex items-center justify-center gap-1.5 rounded py-1.5 transition-colors ${controlMode === id ? "bg-brand-600 text-white" : "text-zinc-400 hover:text-zinc-100"}`}
          >
            <Icon size={13} /> {label}
          </button>
        ))}
      </div>

      {controlMode === "manual" ? (
        <div className="space-y-1.5">
          {STATE_NAMES.map((name) => {
            const s = STATES[name];
            const active = name === cognitiveState;
            return (
              <button
                key={name}
                onClick={() => onManualState(name)}
                className={`flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left ring-1 transition-colors ${
                  active ? TONE[s.tone].btn : "ring-shell-700 text-zinc-300 hover:bg-shell-800"
                }`}
              >
                <kbd className="rounded bg-shell-700 px-1.5 text-[10px] text-zinc-300">{s.key}</kbd>
                <span className="flex-1">
                  <span className="block text-[13px] font-medium">{name}</span>
                  <span className="block text-[11px] opacity-70">{s.mode}</span>
                </span>
                {active && <CircleDot size={14} />}
              </button>
            );
          })}
          <p className="pt-1 text-[11px] text-zinc-500">Shortcuts: 1–4 change state · D toggles this panel</p>
        </div>
      ) : (
        <div className="space-y-3">
          <button
            onClick={onToggleLive}
            disabled={!online}
            className={`flex w-full items-center justify-center gap-2 rounded-md py-2 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
              liveEnabled ? "bg-rose-600 text-white hover:bg-rose-500" : "bg-brand-600 text-white hover:bg-brand-500"
            }`}
          >
            <Camera size={14} />
            {liveEnabled ? "Stop live sensing" : "Start live sensing (webcam + mic)"}
          </button>

          {live.status === "starting" && <p className="text-[11px] text-zinc-400">Requesting camera and microphone…</p>}
          {live.error && (
            <p className="flex items-start gap-1.5 text-[11px] text-rose-300">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" /> {live.error}
            </p>
          )}
          {health?.placeholder_features && health?.mode === "model" && (
            <p className="flex items-start gap-1.5 text-[11px] text-amber-300">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" /> Using placeholder preprocessing. Predictions need Member 1's extract_features functions.
            </p>
          )}

          <div className="space-y-2">
            {Object.entries(DIMENSION_LABELS).map(([dim, label]) => (
              <Bar key={dim} label={label} value={probs?.[dim]} />
            ))}
          </div>
          <p className="text-[11px] text-zinc-500">
            {live.status === "running"
              ? `Sending a 2 s window every 2.5 s${live.latencyMs != null ? ` · ${live.latencyMs} ms` : ""}${
                  live.lastResult?.face_detected === false ? " · no face found" : ""
                }`
              : "Auto mode also follows states pushed to POST /simulate."}
          </p>
        </div>
      )}
    </section>
  );
}
