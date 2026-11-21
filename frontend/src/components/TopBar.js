import { useEffect, useState } from "react";
import {
  Hand, MessageSquare, Mic, MicOff, MonitorUp, PhoneOff, SlidersHorizontal, Users, Video, VideoOff,
} from "lucide-react";
import StatusPill from "./StatusPill";

function useElapsed() {
  const [start] = useState(() => Date.now() - 12 * 60 * 1000 - 7000); // meeting "started" 12 min ago
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const s = Math.floor((now - start) / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function Ctrl({ label, active, onClick, children, badge, className = "" }) {
  return (
    <button onClick={onClick} aria-pressed={active} className={`ctrl-btn relative ${active ? "ctrl-btn-active" : ""} ${className}`}>
      {children}
      <span>{label}</span>
      {badge ? (
        <span className="absolute right-1 top-0.5 min-w-[16px] rounded-full bg-rose-500 px-1 text-[9px] font-semibold leading-4 text-white">
          {badge > 9 ? "9+" : badge}
        </span>
      ) : null}
    </button>
  );
}

export default function TopBar({
  cognitiveState, source, micOn, camOn, sharing, handRaised, chatOpen, chatBadge, demoOpen, minimal,
  onToggleMic, onToggleCam, onToggleShare, onToggleHand, onToggleChat, onToggleDemo, onLeave,
}) {
  const elapsed = useElapsed();
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-shell-700/60 bg-shell-900 px-4">
      <div className="flex min-w-0 items-center gap-3">
        <h1 className="truncate text-sm font-semibold">Project Review Meeting</h1>
        <span className="font-mono text-xs tabular-nums text-zinc-400">{elapsed}</span>
        <StatusPill state={cognitiveState} source={source} />
      </div>

      <div className="flex items-center gap-0.5">
        {/* Hidden in Focus / Deep focus: only Camera, Mic and Share stay */}
        {!minimal && (
          <>
            <Ctrl label="Chat" active={chatOpen} onClick={onToggleChat} badge={chatBadge}>
              <MessageSquare size={18} />
            </Ctrl>
            <Ctrl label="People">
              <Users size={18} />
            </Ctrl>
            <Ctrl label="Raise" active={handRaised} onClick={onToggleHand}>
              <Hand size={18} />
            </Ctrl>
            <Ctrl label="Adapt" active={demoOpen} onClick={onToggleDemo}>
              <SlidersHorizontal size={18} />
            </Ctrl>

            <span className="mx-2 h-8 w-px bg-shell-700" />
          </>
        )}

        <Ctrl label="Camera" active={camOn} onClick={onToggleCam}>
          {camOn ? <Video size={18} /> : <VideoOff size={18} />}
        </Ctrl>
        <Ctrl label={micOn ? "Mic" : "Unmute"} active={micOn} onClick={onToggleMic}>
          {micOn ? <Mic size={18} /> : <MicOff size={18} />}
        </Ctrl>
        <Ctrl label="Share" active={sharing} onClick={onToggleShare}>
          <MonitorUp size={18} />
        </Ctrl>

        <button
          onClick={onLeave}
          className="ml-2 flex items-center gap-1.5 rounded-md bg-rose-600 px-3.5 py-2 text-sm font-semibold text-white transition-colors hover:bg-rose-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-300"
        >
          <PhoneOff size={16} /> Leave
        </button>
      </div>
    </header>
  );
}
