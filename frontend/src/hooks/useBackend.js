import { useEffect, useRef, useState } from "react";
import { API_BASE, WS_URL } from "../config";

/**
 * Keeps a WebSocket open to the FastAPI backend (auto-reconnecting) and polls
 * /health so the UI knows whether the backend is up and in model or mock mode.
 * Every state the backend publishes is passed to onState.
 */
export default function useBackend(onState) {
  const [health, setHealth] = useState(null);
  const [online, setOnline] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const onStateRef = useRef(onState);
  onStateRef.current = onState;

  // Health polling
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
        if (!res.ok) throw new Error(res.statusText);
        const body = await res.json();
        if (!cancelled) {
          setHealth(body);
          setOnline(true);
        }
      } catch {
        if (!cancelled) setOnline(false);
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // WebSocket with exponential back-off reconnect
  useEffect(() => {
    let ws;
    let retry = 0;
    let retryTimer;
    let pingTimer;
    let closed = false;

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => {
        retry = 0;
        setWsConnected(true);
        pingTimer = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "state") onStateRef.current?.(msg);
        } catch {
          /* ignore malformed messages */
        }
      };
      ws.onclose = () => {
        setWsConnected(false);
        clearInterval(pingTimer);
        schedule();
      };
      ws.onerror = () => ws.close();
    };
    const schedule = () => {
      if (closed) return;
      retryTimer = setTimeout(connect, Math.min(10000, 1000 * 2 ** retry++));
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retryTimer);
      clearInterval(pingTimer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  return { health, online, wsConnected };
}

export async function postSimulate(state) {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state }),
  });
  if (!res.ok) throw new Error(`simulate failed: ${res.status}`);
  return res.json();
}
