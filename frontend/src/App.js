import { useCallback, useEffect, useRef, useState } from "react";
import { MoonStar, RotateCcw } from "lucide-react";

import { NORMAL, STATES, STATE_NAMES, TONE } from "./config";
import useBackend from "./hooks/useBackend";
import useLiveInference from "./hooks/useLiveInference";
import useMeetingSim from "./hooks/useMeetingSim";
import useCameraPreview from "./hooks/useCameraPreview";

import LeftRail from "./components/LeftRail";
import TopBar from "./components/TopBar";
import VideoGrid from "./components/VideoGrid";
import SidePanel from "./components/SidePanel";
import Toasts from "./components/Toasts";
import DemoPanel from "./components/DemoPanel";

const TOAST_MS = 5000;

export default function App() {
  // ---- cognitive state (the core of the plan's App.js) -----------------------
  const [cognitiveState, setCognitiveState] = useState(NORMAL);
  const [source, setSource] = useState("manual"); // manual | model | mock
  const isFocusMode = cognitiveState !== NORMAL; // same rule as the plan
  const adapt = STATES[cognitiveState].adapt;

  const [controlMode, setControlMode] = useState("manual"); // manual | auto
  const [lastPrediction, setLastPrediction] = useState(null);

  // ---- meeting UI state ------------------------------------------------------
  const [chatOverride, setChatOverride] = useState(null); // user's own open/close beats the policy until the state changes
  const minimal = Boolean(adapt.minimal); // Focus / Deep focus: no side panels, essential controls only
  const chatOpen = minimal ? false : chatOverride ?? adapt.chatOpen;
  const [panelTab, setPanelTab] = useState("chat");
  const [demoOpen, setDemoOpen] = useState(true);
  const [micOn, setMicOn] = useState(false);
  const [camOn, setCamOn] = useState(false);
  const [sharing, setSharing] = useState(false);
  const [handRaised, setHandRaised] = useState(false);
  const [left, setLeft] = useState(false);
  const [liveEnabled, setLiveEnabled] = useState(false);

  // ---- notifications ---------------------------------------------------------
  const [toasts, setToasts] = useState([]);
  const [heldCount, setHeldCount] = useState(0);
  const [unread, setUnread] = useState({ chat: 0, activity: 0 });

  // Refs so timer/WebSocket callbacks always see the latest values
  const stateRef = useRef(cognitiveState);
  const heldRef = useRef([]);
  const controlModeRef = useRef(controlMode);
  const panelRef = useRef({ chatOpen, panelTab });
  controlModeRef.current = controlMode;
  panelRef.current = { chatOpen, panelTab };

  const pushToast = useCallback((toast) => {
    setToasts((t) => [...t.slice(-2), toast]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== toast.id)), toast.kind === "summary" ? TOAST_MS * 2 : TOAST_MS);
  }, []);

  /** Single entry point for every state change (manual, model, mock, /simulate). */
  const applyState = useCallback(
    (next, src) => {
      if (!STATES[next]) return;
      setSource(src);
      const prev = stateRef.current;
      if (next === prev) return;
      stateRef.current = next;
      setCognitiveState(next);
      setChatOverride(null);
      if (next !== NORMAL) {
        setToasts([]); // clear visible toasts when focus starts
      } else if (heldRef.current.length) {
        const n = heldRef.current.length;
        pushToast({ id: `summary-${Date.now()}`, kind: "summary", text: `${n} notification${n > 1 ? "s" : ""} waiting for you` });
        heldRef.current = [];
        setHeldCount(0);
      }
    },
    [pushToast]
  );

  // ---- backend: WebSocket pushes ---------------------------------------------
  const backend = useBackend(
    useCallback(
      (msg) => {
        if (msg.source === "model" || msg.source === "mock") setLastPrediction(msg);
        if (controlModeRef.current === "auto" && msg.source !== "init") applyState(msg.cognitive_state, msg.source);
      },
      [applyState]
    )
  );

  // ---- live sensing (Option A) -----------------------------------------------
  const live = useLiveInference(liveEnabled && !left, (result) => {
    setLastPrediction(result);
    // The WebSocket normally delivers this too; apply directly only if it is down.
    if (controlModeRef.current === "auto" && !backend.wsConnected) applyState(result.cognitive_state, result.source);
  });
  const previewStream = useCameraPreview(camOn && !liveEnabled && !left);
  const selfStream = liveEnabled ? live.stream : previewStream;

  // ---- simulated meeting activity ---------------------------------------------
  const onNotify = useCallback(
    (n) => {
      const { chatOpen: open, panelTab: tab } = panelRef.current;
      const visible = open && ((n.kind === "chat" && tab === "chat") || (n.kind === "activity" && tab === "activity"));
      if (!visible) setUnread((u) => ({ ...u, [n.kind]: u[n.kind] + 1 }));
      if (visible) return;
      if (STATES[stateRef.current].adapt.toasts) {
        pushToast({ ...n, id: `t-${n.id}` });
      } else {
        heldRef.current.push(n);
        setHeldCount(heldRef.current.length);
      }
    },
    [pushToast]
  );
  const sim = useMeetingSim({ running: !left, onNotify });

  // Opening a tab marks it read
  useEffect(() => {
    if (!chatOpen) return;
    if (panelTab === "chat") setUnread((u) => ({ ...u, chat: 0 }));
    if (panelTab === "activity") {
      setUnread((u) => ({ ...u, activity: 0 }));
      sim.markActivityRead();
    }
  }, [chatOpen, panelTab, sim.markActivityRead]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- handlers ---------------------------------------------------------------
  const setManualState = useCallback(
    (name) => {
      setControlMode("manual");
      setLiveEnabled(false);
      applyState(name, "manual");
    },
    [applyState]
  );

  const changeControlMode = (mode) => {
    setControlMode(mode);
    if (mode === "manual") setLiveEnabled(false);
  };

  const toggleLive = () => {
    setLiveEnabled((on) => {
      if (!on) {
        setCamOn(true);
        setMicOn(true);
      }
      return !on;
    });
  };

  const openPanel = (tab) => {
    setPanelTab(tab);
    setChatOverride(true);
  };

  // Keyboard shortcuts for the live presentation
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      const idx = Number(e.key) - 1;
      if (idx >= 0 && idx < STATE_NAMES.length) setManualState(STATE_NAMES[idx]);
      else if (e.key.toLowerCase() === "d") setDemoOpen((o) => !o);
      else if (e.key.toLowerCase() === "c" && !STATES[stateRef.current].adapt.minimal) setChatOverride(!chatOpen);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setManualState, chatOpen]);

  // ---- render -----------------------------------------------------------------
  if (left) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-shell-950 text-zinc-200">
        <p className="text-lg">You left the meeting</p>
        <button onClick={() => setLeft(false)} className="flex items-center gap-2 rounded-md bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-500">
          <RotateCcw size={15} /> Rejoin
        </button>
      </div>
    );
  }

  const s = STATES[cognitiveState];
  const badgeChat = adapt.badges && !chatOpen ? unread.chat : 0;

  return (
    <div className="flex h-screen overflow-hidden bg-shell-950 font-sans text-zinc-100">
      {/* Left rail slides away in minimal modes */}
      <div
        className={`shrink-0 overflow-hidden transition-[width] duration-500 ease-in-out ${minimal ? "w-0" : "w-[68px]"}`}
        aria-hidden={minimal}
      >
        <LeftRail
          badges={unread}
          showBadges={adapt.badges}
          focusActive={isFocusMode}
          onSelect={(id) => (id === "chat" || id === "activity") && openPanel(id)}
        />
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          cognitiveState={cognitiveState}
          source={source}
          micOn={micOn}
          camOn={camOn}
          sharing={sharing}
          handRaised={handRaised}
          chatOpen={chatOpen}
          chatBadge={badgeChat}
          demoOpen={demoOpen}
          minimal={minimal}
          onToggleMic={() => setMicOn((v) => !v)}
          onToggleCam={() => setCamOn((v) => !v)}
          onToggleShare={() => setSharing((v) => !v)}
          onToggleHand={() => setHandRaised((v) => !v)}
          onToggleChat={() => setChatOverride(!chatOpen)}
          onToggleDemo={() => setDemoOpen((v) => !v)}
          onLeave={() => {
            setLiveEnabled(false);
            setLeft(true);
          }}
        />

        <div className="flex min-h-0 flex-1">
          <main className="relative flex min-w-0 flex-1 flex-col">
            {/* Explains the adaptation to the user (and the professor) */}
            <div
              className={`overflow-hidden transition-all duration-500 ${isFocusMode ? "max-h-14 opacity-100" : "max-h-0 opacity-0"}`}
              aria-hidden={!isFocusMode}
            >
              <div className="mx-4 mt-3 flex items-center gap-3 rounded-lg bg-shell-850 px-4 py-2 text-xs ring-1 ring-shell-700">
                <MoonStar size={14} className="shrink-0 text-brand-300" />
                <span className="min-w-0 flex-1 truncate text-zinc-300">
                  <span className={`mr-1.5 rounded px-1.5 py-0.5 font-semibold ring-1 ${TONE[s.tone].pill}`}>{s.mode}</span>
                  {s.description}
                  {heldCount > 0 && <span className="ml-1 text-zinc-400">· {heldCount} held</span>}
                </span>
                {!chatOpen && !minimal && (
                  <button onClick={() => openPanel("chat")} className="shrink-0 text-brand-300 hover:text-brand-200">
                    Show chat
                  </button>
                )}
              </div>
            </div>

            <div className="min-h-0 flex-1">
              <VideoGrid
                layout={adapt.layout}
                activeSpeaker={sim.activeSpeaker}
                stream={selfStream}
                camOn={camOn}
                micOn={micOn}
                handRaised={handRaised}
                sharing={sharing}
              />
            </div>

            <DemoPanel
              open={demoOpen}
              onClose={() => setDemoOpen(false)}
              controlMode={controlMode}
              onControlMode={changeControlMode}
              cognitiveState={cognitiveState}
              onManualState={setManualState}
              backend={backend}
              live={live}
              liveEnabled={liveEnabled}
              onToggleLive={toggleLive}
              lastPrediction={lastPrediction}
            />
          </main>

          <SidePanel
            open={chatOpen}
            tab={panelTab}
            onTab={setPanelTab}
            onClose={() => setChatOverride(false)}
            messages={sim.messages}
            activity={sim.activity}
            onSend={sim.sendMessage}
          />
        </div>
      </div>

      <Toasts
        toasts={toasts}
        rightOffset={chatOpen ? 336 : 16}
        onDismiss={(id) => setToasts((t) => t.filter((x) => x.id !== id))}
        onOpen={(t) => {
          openPanel(t.kind === "chat" ? "chat" : "activity");
          setToasts((list) => list.filter((x) => x.id !== t.id));
        }}
      />
    </div>
  );
}
