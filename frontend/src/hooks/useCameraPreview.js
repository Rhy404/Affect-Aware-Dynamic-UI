import { useEffect, useState } from "react";

/** Plain camera preview for the self tile when live sensing is off. */
export default function useCameraPreview(enabled) {
  const [stream, setStream] = useState(null);
  useEffect(() => {
    if (!enabled) return;
    let media;
    let cancelled = false;
    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        media = s;
        setStream(s);
      })
      .catch(() => setStream(null));
    return () => {
      cancelled = true;
      media?.getTracks().forEach((t) => t.stop());
      setStream(null);
    };
  }, [enabled]);
  return stream;
}
