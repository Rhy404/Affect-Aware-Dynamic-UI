import { Bell, CalendarDays, MessageSquare, MoonStar, Users, CircleHelp } from "lucide-react";

const ITEMS = [
  { id: "activity", label: "Activity", Icon: Bell },
  { id: "chat", label: "Chat", Icon: MessageSquare },
  { id: "teams", label: "Teams", Icon: Users },
  { id: "calendar", label: "Calendar", Icon: CalendarDays },
];

export default function LeftRail({ badges, showBadges, focusActive, onSelect }) {
  return (
    <nav aria-label="App navigation" className="flex w-[68px] shrink-0 flex-col items-center border-r border-shell-700/60 bg-shell-900 py-3">
      <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-sm font-bold text-white shadow-lg shadow-brand-600/30" title="AffectMeet">
        A
      </div>

      <ul className="flex flex-1 flex-col items-center gap-1">
        {ITEMS.map(({ id, label, Icon }) => {
          const count = badges[id] || 0;
          const active = id === "teams";
          return (
            <li key={id}>
              <button
                onClick={() => onSelect?.(id)}
                className={`group relative flex w-14 flex-col items-center gap-1 rounded-md py-2 text-[10px] transition-colors ${
                  active ? "text-brand-300" : "text-zinc-400 hover:text-zinc-100"
                }`}
                aria-label={count && showBadges ? `${label}, ${count} unread` : label}
              >
                {active && <span className="absolute left-0 top-2 h-7 w-[3px] rounded-r bg-brand-400" />}
                <span className="relative">
                  <Icon size={20} strokeWidth={active ? 2.2 : 1.8} />
                  <span
                    className={`absolute -right-2.5 -top-1.5 min-w-[18px] rounded-full bg-rose-500 px-1 text-center text-[10px] font-semibold leading-[18px] text-white transition-all duration-300 ${
                      count && showBadges ? "scale-100 opacity-100" : "scale-50 opacity-0"
                    }`}
                  >
                    {count > 9 ? "9+" : count}
                  </span>
                </span>
                {label}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="flex flex-col items-center gap-3 text-zinc-500">
        <span
          title="Notifications are being held"
          className={`transition-opacity duration-300 ${focusActive ? "opacity-100 text-brand-300" : "opacity-0"}`}
        >
          <MoonStar size={18} />
        </span>
        <CircleHelp size={18} />
      </div>
    </nav>
  );
}
