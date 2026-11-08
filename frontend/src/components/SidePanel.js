import { useEffect, useRef, useState } from "react";
import { AtSign, CalendarClock, FileText, Heart, Send, X } from "lucide-react";
import { PARTICIPANTS } from "../hooks/useMeetingSim";

const byId = Object.fromEntries(PARTICIPANTS.map((p) => [p.id, p]));
const ACT_ICON = { mention: AtSign, reaction: Heart, file: FileText, calendar: CalendarClock };

function ChatTab({ messages, onSend }) {
  const [draft, setDraft] = useState("");
  const endRef = useRef(null);
  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and
    // React would try to call a returned value as the effect's cleanup.
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  const submit = (e) => {
    e.preventDefault();
    if (draft.trim()) onSend(draft.trim());
    setDraft("");
  };

  return (
    <>
      <ol className="scroll-thin flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.map((m) => {
          const p = byId[m.from];
          const mine = p.self;
          return (
            <li key={m.id} className={`flex gap-2.5 ${mine ? "flex-row-reverse" : ""}`}>
              {!mine && (
                <span className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br text-[10px] font-semibold text-white ${p.color}`}>
                  {p.initials}
                </span>
              )}
              <div className={`max-w-[80%] ${mine ? "text-right" : ""}`}>
                <div className="mb-0.5 text-[11px] text-zinc-500">
                  {!mine && <span className="mr-1.5 font-medium text-zinc-300">{p.name}</span>}
                  {m.time}
                </div>
                <p className={`inline-block rounded-md px-3 py-1.5 text-left text-[13px] leading-snug ${mine ? "bg-brand-600/40 text-zinc-100" : "bg-shell-700 text-zinc-200"}`}>
                  {m.text}
                </p>
              </div>
            </li>
          );
        })}
        <li ref={endRef} />
      </ol>
      <form onSubmit={submit} className="border-t border-shell-700/60 p-3">
        <div className="flex items-center gap-2 rounded-md bg-shell-800 px-3 py-2 ring-1 ring-shell-700 focus-within:ring-brand-500">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Type a message"
            aria-label="Type a message"
            className="min-w-0 flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-500 outline-none"
          />
          <button type="submit" aria-label="Send" className="text-brand-300 hover:text-brand-200 disabled:opacity-40" disabled={!draft.trim()}>
            <Send size={16} />
          </button>
        </div>
      </form>
    </>
  );
}

function ActivityTab({ activity }) {
  if (!activity.length) return <p className="px-4 py-6 text-sm text-zinc-500">No activity yet.</p>;
  return (
    <ul className="scroll-thin flex-1 divide-y divide-shell-700/50 overflow-y-auto">
      {activity.map((a) => {
        const Icon = ACT_ICON[a.icon] || AtSign;
        return (
          <li key={a.id} className={`flex gap-3 px-4 py-3 text-[13px] ${a.unread ? "bg-brand-600/10" : ""}`}>
            <Icon size={16} className="mt-0.5 shrink-0 text-brand-300" />
            <div className="min-w-0">
              <p className="text-zinc-200">{a.text}</p>
              <p className="text-[11px] text-zinc-500">{a.time}</p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/** Right-hand panel. Width animates to 0 when closed (Focus Mode). */
export default function SidePanel({ open, tab, onTab, onClose, messages, activity, onSend }) {
  return (
    <aside
      aria-label="Meeting chat and activity"
      aria-hidden={!open}
      className={`shrink-0 overflow-hidden border-l border-shell-700/60 bg-shell-900 transition-[width,opacity] duration-500 ease-in-out ${
        open ? "w-80 opacity-100" : "w-0 border-l-0 opacity-0"
      }`}
    >
      <div className="flex h-full w-80 flex-col">
        <div className="flex h-12 items-center justify-between border-b border-shell-700/60 px-4">
          <div className="flex gap-4 text-sm" role="tablist">
            {["chat", "activity"].map((t) => (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => onTab(t)}
                className={`border-b-2 py-3 capitalize transition-colors ${tab === t ? "border-brand-400 font-semibold text-white" : "border-transparent text-zinc-400 hover:text-zinc-200"}`}
              >
                {t === "chat" ? "Meeting chat" : "Activity"}
              </button>
            ))}
          </div>
          <button onClick={onClose} aria-label="Close panel" className="rounded p-1 text-zinc-400 hover:bg-shell-700 hover:text-white">
            <X size={16} />
          </button>
        </div>
        {tab === "chat" ? <ChatTab messages={messages} onSend={onSend} /> : <ActivityTab activity={activity} />}
      </div>
    </aside>
  );
}
