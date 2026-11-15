import { useEffect, useRef } from "react";
import { Hand, MicOff, MonitorUp } from "lucide-react";
import { PARTICIPANTS } from "../hooks/useMeetingSim";

function SelfVideo({ stream }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.srcObject = stream || null;
  }, [stream]);
  return <video ref={ref} autoPlay muted playsInline className="h-full w-full scale-x-[-1] object-cover" />;
}

function Tile({ p, speaking, stream, camOn, micOn, handRaised, size = "md" }) {
  const showVideo = p.self && camOn && stream;
  const avatar = size === "lg" ? "h-28 w-28 text-4xl" : size === "sm" ? "h-10 w-10 text-sm" : "h-20 w-20 text-2xl";
  return (
    <div
      className={`relative flex h-full w-full items-center justify-center overflow-hidden rounded-lg bg-shell-800 transition-shadow duration-300 ${
        speaking ? "ring-2 ring-brand-400 animate-pulse-ring" : "ring-1 ring-shell-700/60"
      }`}
    >
      {showVideo ? (
        <SelfVideo stream={stream} />
      ) : (
        <div className={`flex items-center justify-center rounded-full bg-gradient-to-br font-semibold text-white shadow-lg ${p.color} ${avatar}`}>
          {p.initials}
        </div>
      )}
      <div className="absolute bottom-2 left-2 flex max-w-[85%] items-center gap-1.5 rounded bg-black/55 px-2 py-0.5 text-xs text-white backdrop-blur">
        {p.self && !micOn && <MicOff size={12} className="text-rose-300" />}
        {p.self && handRaised && <Hand size={12} className="text-amber-300" />}
        <span className="truncate">{p.name}</span>
      </div>
    </div>
  );
}

function ScreenShare() {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 rounded-lg bg-gradient-to-br from-shell-800 to-shell-850 ring-1 ring-brand-500/40">
      <MonitorUp className="text-brand-300" size={36} />
      <p className="text-sm text-zinc-300">You are sharing your screen</p>
    </div>
  );
}

/**
 * layout: "grid" (gallery) | "spotlight" (speaker + filmstrip) | "speaker-only"
 */
export default function VideoGrid({ layout, activeSpeaker, stream, camOn, micOn, handRaised, sharing }) {
  const speaker = PARTICIPANTS.find((p) => p.id === activeSpeaker) || PARTICIPANTS[1];
  const tileProps = (p) => ({ p, speaking: p.id === activeSpeaker, stream, camOn, micOn, handRaised });

  if (layout === "grid") {
    return (
      <div key="grid" className="grid h-full animate-toast-in grid-cols-2 gap-3 p-4 lg:grid-cols-3" style={{ gridAutoRows: "1fr" }}>
        {sharing && (
          <div className="col-span-2 lg:col-span-3">
            <ScreenShare />
          </div>
        )}
        {PARTICIPANTS.map((p) => (
          <Tile key={p.id} {...tileProps(p)} />
        ))}
      </div>
    );
  }

  const others = PARTICIPANTS.filter((p) => p.id !== speaker.id);
  return (
    <div key={layout} className="flex h-full animate-toast-in flex-col gap-3 p-4">
      <div className="min-h-0 flex-1">{sharing ? <ScreenShare /> : <Tile {...tileProps(speaker)} size="lg" />}</div>
      <div
        className={`grid shrink-0 grid-cols-5 gap-3 overflow-hidden transition-all duration-500 ${
          layout === "spotlight" ? "h-24 opacity-100" : "h-0 opacity-0"
        }`}
        aria-hidden={layout !== "spotlight"}
      >
        {others.map((p) => (
          <Tile key={p.id} {...tileProps(p)} size="sm" />
        ))}
      </div>
    </div>
  );
}
