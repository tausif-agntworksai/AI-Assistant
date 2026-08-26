import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Consent } from "./screens/Consent";
import { applyStoredTheme } from "./useTheme";
import "./theme.css";

// Before the first paint, so there is no flash of the wrong palette.
applyStoredTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Consent />
  </StrictMode>
);
