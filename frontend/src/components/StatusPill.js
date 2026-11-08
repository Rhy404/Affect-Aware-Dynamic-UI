import { STATES, TONE } from "../config";

const SOURCE_LABEL = { model: "model", mock: "mock model", manual: "manual", init: "manual" };

export default function StatusPill({ state, source }) {
  const s = STATES[state];
  const tone = TONE[s.tone];
  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ring-1 transition-colors duration-500 ${tone.pill}`}
      title={s.description}
    >
      <span className={`h-2 w-2 rounded-full ${tone.dot} ${state !== "Normal Load" ? "animate-pulse" : ""}`} />
      {state}
      <span className="text-[10px] font-normal opacity-70">· {SOURCE_LABEL[source] || source}</span>
    </span>
  );
}
