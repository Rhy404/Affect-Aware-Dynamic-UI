import { AtSign, BellRing, MessageSquare, X } from "lucide-react";
import { PARTICIPANTS } from "../hooks/useMeetingSim";

const byId = Object.fromEntries(PARTICIPANTS.map((p) => [p.id, p]));

/** Bottom-right notification toasts (only shown when the state allows). */
export default function Toasts({ toasts, onDismiss, onOpen, rightOffset }) {
  return (
    <div
      className="pointer-events-none fixed bottom-4 z-40 flex w-80 flex-col gap-2 transition-[right] duration-500"
      style={{ right: rightOffset }}
      aria-live="polite"
    >
      {toasts.map((t) => {
        const summary = t.kind === "summary";
        const Icon = summary ? BellRing : t.kind === "chat" ? MessageSquare : AtSign;
        const title = summary ? "While you were focused" : t.kind === "chat" ? byId[t.from]?.name : "Activity";
        return (
          <div
            key={t.id}
            className={`pointer-events-auto flex animate-toast-in gap-3 rounded-lg border p-3 shadow-xl shadow-black/40 ${
              summary ? "border-brand-500/50 bg-shell-800" : "border-shell-700 bg-shell-850"
            }`}
          >
            <Icon size={16} className="mt-0.5 shrink-0 text-brand-300" />
            <button className="min-w-0 flex-1 text-left" onClick={() => onOpen(t)}>
              <p className="text-xs font-semibold text-zinc-100">{title}</p>
              <p className="truncate text-[13px] text-zinc-300">{t.text}</p>
              {summary && <p className="mt-1 text-xs font-medium text-brand-300">Review now →</p>}
            </button>
            <button onClick={() => onDismiss(t.id)} aria-label="Dismiss" className="self-start text-zinc-500 hover:text-zinc-200">
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
