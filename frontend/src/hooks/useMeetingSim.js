import { useCallback, useEffect, useRef, useState } from "react";

// Simulated meeting so the prototype feels alive during the demo:
// rotating active speaker, incoming chat messages and activity notifications.

export const PARTICIPANTS = [
  { id: "you", name: "You", initials: "ST", color: "from-indigo-500 to-violet-600", self: true },
  { id: "priya", name: "Priya Sharma", initials: "PS", color: "from-rose-500 to-orange-400" },
  { id: "arjun", name: "Arjun Mehta", initials: "AM", color: "from-emerald-500 to-teal-500" },
  { id: "neha", name: "Neha Kapoor", initials: "NK", color: "from-sky-500 to-cyan-400" },
  { id: "rahul", name: "Rahul Verma", initials: "RV", color: "from-amber-500 to-yellow-400" },
  { id: "ananya", name: "Ananya Iyer", initials: "AI", color: "from-fuchsia-500 to-pink-500" },
];

const CHAT_SCRIPT = [
  ["priya", "Can everyone see the Q3 roadmap slide?"],
  ["arjun", "Yes, looks good on my end 👍"],
  ["neha", "Quick question: is the API freeze still on the 20th?"],
  ["rahul", "I'll share the latency numbers after this section"],
  ["ananya", "Dropping the design doc link in the channel"],
  ["priya", "@Sanya can you walk us through the dashboard changes next?"],
  ["arjun", "+1, the new focus-mode transitions look really smooth"],
  ["neha", "Let's take the migration plan offline"],
  ["rahul", "Uploaded benchmark_results.xlsx"],
  ["ananya", "Reminder: retro at 5 pm today"],
];

const ACTIVITY_SCRIPT = [
  { icon: "mention", text: "Priya mentioned you in #release-planning" },
  { icon: "reaction", text: "Arjun reacted ❤️ to your message" },
  { icon: "file", text: "Rahul shared benchmark_results.xlsx" },
  { icon: "calendar", text: "Design sync starts in 15 minutes" },
  { icon: "mention", text: "Neha replied to your thread" },
  { icon: "reaction", text: "Ananya reacted 🎉 to the roadmap post" },
];

const SEED_MESSAGES = [
  { id: "m0", from: "neha", text: "Morning all! Agenda is pinned above.", time: offsetTime(-6) },
  { id: "m1", from: "rahul", text: "Joining in a sec", time: offsetTime(-5) },
];

function offsetTime(min) {
  return new Date(Date.now() + min * 60000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
const now = () => offsetTime(0);

/**
 * onNotify(notification) is called for every incoming event; the App decides
 * (based on the cognitive state) whether to show it as a toast or hold it.
 */
export default function useMeetingSim({ running, onNotify }) {
  const [messages, setMessages] = useState(SEED_MESSAGES);
  const [activity, setActivity] = useState([]);
  const [activeSpeaker, setActiveSpeaker] = useState("priya");
  const chatIdx = useRef(0);
  const actIdx = useRef(0);
  const onNotifyRef = useRef(onNotify);
  onNotifyRef.current = onNotify;

  useEffect(() => {
    if (!running) return;
    const speakers = PARTICIPANTS.filter((p) => !p.self).map((p) => p.id);
    const speakerTimer = setInterval(() => {
      setActiveSpeaker((cur) => {
        const options = speakers.filter((s) => s !== cur);
        return options[Math.floor(Math.random() * options.length)];
      });
    }, 7000);

    const chatTimer = setInterval(() => {
      const [from, text] = CHAT_SCRIPT[chatIdx.current++ % CHAT_SCRIPT.length];
      const msg = { id: `c${Date.now()}`, from, text, time: now() };
      setMessages((m) => [...m.slice(-60), msg]);
      onNotifyRef.current?.({ kind: "chat", from, text, id: msg.id });
    }, 11000);

    const activityTimer = setInterval(() => {
      const a = ACTIVITY_SCRIPT[actIdx.current++ % ACTIVITY_SCRIPT.length];
      const item = { ...a, id: `a${Date.now()}`, time: now(), unread: true };
      setActivity((list) => [item, ...list].slice(0, 30));
      onNotifyRef.current?.({ kind: "activity", text: a.text, icon: a.icon, id: item.id });
    }, 17000);

    return () => {
      clearInterval(speakerTimer);
      clearInterval(chatTimer);
      clearInterval(activityTimer);
    };
  }, [running]);

  const sendMessage = useCallback((text) => {
    setMessages((m) => [...m, { id: `u${Date.now()}`, from: "you", text, time: now() }]);
  }, []);

  const markActivityRead = useCallback(() => {
    setActivity((list) => list.map((a) => ({ ...a, unread: false })));
  }, []);

  return { messages, activity, activeSpeaker, sendMessage, markActivityRead };
}
