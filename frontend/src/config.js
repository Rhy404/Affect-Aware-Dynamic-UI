// Central configuration for the adaptive UI.
// The ADAPTATION table is the "policy" of the system: how each predicted
// cognitive-load state changes the interface. Tune it here; nothing else
// needs to change. (Worth citing as a table in the research paper.)

export const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws";

export const NORMAL = "Normal Load";

export const STATES = {
  "Normal Load": {
    key: "1",
    short: "Normal",
    mode: "Normal mode",
    tone: "emerald",
    description: "Full interface: chat open, notifications and badges shown, gallery view.",
    adapt: { chatOpen: true, badges: true, toasts: true, layout: "grid" },
  },
  "High Temporal Demand": {
    key: "2",
    short: "Temporal",
    mode: "Quiet mode",
    tone: "amber",
    description: "Time pressure detected: chat collapses, notifications and badges are held.",
    adapt: { chatOpen: false, badges: false, toasts: false, layout: "grid" },
  },
  "High Mental Demand": {
    key: "3",
    short: "Mental",
    mode: "Focus mode",
    tone: "rose",
    description: "Heavy thinking detected: chat collapses, notifications held, speaker spotlight.",
    adapt: { chatOpen: false, badges: false, toasts: false, layout: "spotlight" },
  },
  "High Effort": {
    key: "4",
    short: "Effort",
    mode: "Deep focus",
    tone: "fuchsia",
    description: "High effort detected: everything non-essential hidden, active speaker only.",
    adapt: { chatOpen: false, badges: false, toasts: false, layout: "speaker-only" },
  },
};

export const STATE_NAMES = Object.keys(STATES);

// Tailwind classes per tone (spelled out so Tailwind's purge keeps them).
export const TONE = {
  emerald: { pill: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30", dot: "bg-emerald-400", bar: "bg-emerald-400", btn: "bg-emerald-500/20 ring-emerald-400 text-emerald-200" },
  amber: { pill: "bg-amber-500/15 text-amber-300 ring-amber-500/30", dot: "bg-amber-400", bar: "bg-amber-400", btn: "bg-amber-500/20 ring-amber-400 text-amber-200" },
  rose: { pill: "bg-rose-500/15 text-rose-300 ring-rose-500/30", dot: "bg-rose-400", bar: "bg-rose-400", btn: "bg-rose-500/20 ring-rose-400 text-rose-200" },
  fuchsia: { pill: "bg-fuchsia-500/15 text-fuchsia-300 ring-fuchsia-500/30", dot: "bg-fuchsia-400", bar: "bg-fuchsia-400", btn: "bg-fuchsia-500/20 ring-fuchsia-400 text-fuchsia-200" },
};

export const DIMENSION_LABELS = {
  mental_demand: "Mental Demand",
  temporal_demand: "Temporal Demand",
  effort: "Effort",
};

// Live sensing: how often the browser sends a webcam + mic window to the backend.
export const LIVE_INTERVAL_MS = 2500;
export const FRAMES_PER_WINDOW = 8;
export const AUDIO_WINDOW_SECONDS = 2;
