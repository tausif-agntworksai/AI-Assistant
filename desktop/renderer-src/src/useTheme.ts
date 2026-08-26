import { useCallback, useEffect, useState } from "react";

export type ThemeChoice = "system" | "light" | "dark";

const STORAGE_KEY = "jarvis.theme";

function read(): ThemeChoice {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark" || saved === "system") return saved;
  } catch {
    /* private window, cleared site data — the default is fine */
  }
  return "system";
}

/**
 * Follows the OS unless the user has overridden it.
 *
 * "system" deliberately removes the attribute rather than resolving it to a
 * value, so `prefers-color-scheme` in the stylesheet keeps working and the
 * window changes with the OS while it is open — no reload, no listener.
 */
export function useTheme(): [ThemeChoice, (next: ThemeChoice) => void] {
  const [choice, setChoice] = useState<ThemeChoice>(read);

  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", choice);
    try {
      localStorage.setItem(STORAGE_KEY, choice);
    } catch {
      /* the choice still applies to this session */
    }
  }, [choice]);

  return [choice, useCallback((next: ThemeChoice) => setChoice(next), [])];
}

/** Applied before React mounts, so there is no flash of the wrong theme. */
export function applyStoredTheme(): void {
  const choice = read();
  if (choice !== "system") document.documentElement.setAttribute("data-theme", choice);
}
