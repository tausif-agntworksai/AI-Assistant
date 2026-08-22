/**
 * Build config for the renderer.
 *
 * Three things here are load-bearing, and two of them fail in a way you only
 * see in a packaged build:
 *
 * **`base: "./"`** — Vite's default emits `<script src="/assets/index-*.js">`,
 * and under `file://` a leading slash resolves to the filesystem root. The page
 * comes up blank. It works in `vite dev` and in any app that loads over HTTP
 * (the sibling AI Calculator does, which is why its config needs none of this),
 * so nothing warns you until the installer is built and the window is white.
 *
 * **Three entry points, not one router.** Each screen is a real page, and a
 * screen change is a real navigation. That is what guarantees the HUD's
 * WebSocket, reconnect timer and level-meter timer are gone when you leave it —
 * a router would make that cleanup something to remember by hand. The main
 * process already serialises navigation through one promise chain, so the cost
 * is a `loadFile` we were doing anyway.
 *
 * **Sourcemaps in production.** The window is frameless with no menu, so there
 * is no way to open DevTools by hand; errors are forwarded to the main process
 * log instead, and without sourcemaps they arrive as `index-a1b2c3.js:1:48120`.
 */
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const here = (...parts: string[]) => resolve(__dirname, "renderer-src", ...parts);

export default defineConfig({
  root: here(),
  base: "./",
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "renderer-dist"),
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        index: here("index.html"),
        auth: here("auth.html"),
        consent: here("consent.html"),
      },
    },
  },
  server: { port: 5273 },
});
