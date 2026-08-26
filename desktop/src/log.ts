/**
 * Diagnostics for the main process, written to a file as well as the console.
 *
 * A packaged Windows app is a GUI-subsystem binary: it has no console, so
 * `console.error` goes nowhere at all. That is fine for noise and fatal for the
 * one thing you actually need after shipping — a renderer that fails to paint.
 * The window is frameless with no menu, so there is no way to open DevTools by
 * hand either, which leaves a white screen with no evidence anywhere.
 *
 * So every diagnostic lands in `desktop.log`, next to the engine's own log and
 * reachable from the tray. Small, append-only, and truncated when it gets silly.
 */

import { appendFileSync, mkdirSync, renameSync, statSync } from "node:fs";
import path from "node:path";
import { app } from "electron";

const MAX_BYTES = 512 * 1024;

let target: string | null = null;

function file(): string {
  if (target) return target;
  const dir = app.getPath("logs");
  try {
    mkdirSync(dir, { recursive: true });
  } catch {
    /* the console half still works */
  }
  target = path.join(dir, "desktop.log");
  return target;
}

export function logPath(): string {
  return file();
}

function rotateIfLarge(): void {
  try {
    if (statSync(file()).size > MAX_BYTES) {
      renameSync(file(), `${file()}.1`);
    }
  } catch {
    /* no file yet, or it is in use — either way, keep going */
  }
}

/** Writes one line. Never throws: logging must not be able to break a launch. */
export function log(level: "info" | "warn" | "error", message: string): void {
  const line = `${new Date().toISOString()} ${level.toUpperCase()} ${message}\n`;
  if (level === "error") console.error(line.trimEnd());
  else if (level === "warn") console.warn(line.trimEnd());
  else console.log(line.trimEnd());

  try {
    rotateIfLarge();
    appendFileSync(file(), line, "utf8");
  } catch {
    /* nothing sensible to do if even the log is unwritable */
  }
}
