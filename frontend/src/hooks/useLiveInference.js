import { useEffect, useRef, useState } from "react";
import { API_BASE, AUDIO_WINDOW_SECONDS, FRAMES_PER_WINDOW, LIVE_INTERVAL_MS } from "../config";

const TARGET_SR = 16000;

/**
 * Live sensing (Plan section D, Option A).
 * While `enabled`, captures the webcam + microphone and every LIVE_INTERVAL_MS
 * sends the latest window (JPEG frames + 16 kHz WAV) to POST /predict-live.
 */
export default function useLiveInference(enabled, onResult) {
  const [status, setStatus] = useState("idle"); // idle | starting | running | error
  const [error, setError] = useState(null);
  const [stream, setStream] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  useEffect(() => {
    if (!enabled) {
      setStatus("idle");
      return;
    }
    let cancelled = false;
    let media, audioCtx, processor, frameTimer, sendTimer;
    let frames = [];
    let audioChunks = [];
    let inFlight = false;

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const ctx2d = canvas.getContext("2d");

    const start = async () => {
      setStatus("starting");
      setError(null);
      try {
        media = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch (e) {
        if (!cancelled) {
          setStatus("error");
          setError(e.name === "NotAllowedError" ? "Camera/microphone permission denied" : e.message);
        }
        return;
      }
      if (cancelled) return media.getTracks().forEach((t) => t.stop());
      setStream(media);
      video.srcObject = media;
      await video.play().catch(() => {});

      // Audio: buffer raw PCM samples
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioCtx();
      const source = audioCtx.createMediaStreamSource(media);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const maxSamples = audioCtx.sampleRate * AUDIO_WINDOW_SECONDS;
      processor.onaudioprocess = (e) => {
        audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        let total = audioChunks.reduce((n, c) => n + c.length, 0);
        while (total - audioChunks[0].length >= maxSamples) total -= audioChunks.shift().length;
      };
      const silent = audioCtx.createGain();
      silent.gain.value = 0;
      source.connect(processor);
      processor.connect(silent);
      silent.connect(audioCtx.destination);

      // Video: grab frames evenly across the window
      const frameEvery = (AUDIO_WINDOW_SECONDS * 1000) / FRAMES_PER_WINDOW;
      frameTimer = setInterval(() => {
        if (video.readyState < 2) return;
        ctx2d.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((b) => {
          if (!b) return;
          frames.push(b);
          if (frames.length > FRAMES_PER_WINDOW) frames.shift();
        }, "image/jpeg", 0.85);
      }, frameEvery);

      sendTimer = setInterval(send, LIVE_INTERVAL_MS);
      setStatus("running");
    };

    const send = async () => {
      if (inFlight || cancelled || (!frames.length && !audioChunks.length)) return;
      inFlight = true;
      const form = new FormData();
      frames.forEach((b, i) => form.append("frames", b, `frame_${i}.jpg`));
      if (audioChunks.length && audioCtx) {
        const wav = encodeWav(resample(concat(audioChunks), audioCtx.sampleRate, TARGET_SR), TARGET_SR);
        form.append("audio", wav, "window.wav");
      }
      const t0 = performance.now();
      try {
        const res = await fetch(`${API_BASE}/predict-live`, { method: "POST", body: form });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || res.statusText);
        if (cancelled) return;
        setLatencyMs(Math.round(performance.now() - t0));
        setLastResult(body);
        setError(null);
        onResultRef.current?.(body);
      } catch (e) {
        if (!cancelled) setError(`Backend: ${e.message}`);
      } finally {
        inFlight = false;
      }
    };

    start();
    return () => {
      cancelled = true;
      clearInterval(frameTimer);
      clearInterval(sendTimer);
      processor && (processor.onaudioprocess = null);
      audioCtx?.close().catch(() => {});
      media?.getTracks().forEach((t) => t.stop());
      setStream(null);
    };
  }, [enabled]);

  return { status, error, stream, lastResult, latencyMs };
}

// ---------------------------------------------------------------- audio helpers

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    out[i] = input[i0] + (input[i1] - input[i0]) * (pos - i0); // linear interpolation
  }
  return out;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buffer);
  const str = (off, s) => [...s].forEach((ch, i) => v.setUint8(off + i, ch.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true); // PCM chunk size
  v.setUint16(20, 1, true); // PCM format
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
