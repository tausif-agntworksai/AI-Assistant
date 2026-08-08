/**
 * Generates assets/icon.png and build/icon.ico from an inline SVG.
 *
 * Run with Electron (`npm run icon`), not plain Node — it uses Chromium to
 * rasterise the SVG, so there's no image dependency to install.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, nativeImage } from "electron";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <radialGradient id="bg" cx="35%" cy="28%">
      <stop offset="0%" stop-color="#1d2b42"/>
      <stop offset="100%" stop-color="#080c14"/>
    </radialGradient>
    <radialGradient id="core" cx="38%" cy="32%">
      <stop offset="0%" stop-color="#b6f4ff"/>
      <stop offset="55%" stop-color="#35d6ff"/>
      <stop offset="100%" stop-color="#0d7ea0"/>
    </radialGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="20" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="512" height="512" rx="112" fill="url(#bg)"/>
  <circle cx="256" cy="256" r="150" fill="none" stroke="#1b7f99" stroke-width="3" opacity=".55"/>
  <circle cx="256" cy="256" r="186" fill="none" stroke="#1b7f99" stroke-width="2" opacity=".3"/>
  <g filter="url(#glow)">
    <circle cx="256" cy="256" r="92" fill="url(#core)"/>
  </g>
  <g stroke="#35d6ff" stroke-width="10" stroke-linecap="round" opacity=".9">
    <line x1="256" y1="86" x2="256" y2="126"/>
    <line x1="256" y1="386" x2="256" y2="426"/>
    <line x1="86" y1="256" x2="126" y2="256"/>
    <line x1="386" y1="256" x2="426" y2="256"/>
  </g>
</svg>`;

const SIZES = [16, 24, 32, 48, 64, 128, 256];

async function render(win, size) {
  const dataUrl = `data:image/svg+xml;base64,${Buffer.from(SVG).toString("base64")}`;
  await win.loadURL(
    `data:text/html,<body style="margin:0;background:transparent">` +
      `<img src="${dataUrl}" width="${size}" height="${size}">` +
      `</body>`
  );
  // Offscreen windows still composite frames, but the first one lands a beat
  // after load resolves; capturing immediately yields a blank image.
  await new Promise((r) => setTimeout(r, 120));
  return win.webContents.capturePage({ x: 0, y: 0, width: size, height: size });
}

app.whenReady().then(async () => {
  // `offscreen: true` is what makes capturePage() work on a window that is
  // never shown — on a plain hidden window the promise simply never settles.
  const win = new BrowserWindow({
    width: 512,
    height: 512,
    show: false,
    webPreferences: { offscreen: true },
  });

  mkdirSync(path.join(root, "assets"), { recursive: true });
  mkdirSync(path.join(root, "build"), { recursive: true });

  const big = await render(win, 256);
  writeFileSync(path.join(root, "assets", "icon.png"), big.toPNG());

  // electron-builder needs a real multi-size .ico; nativeImage can only write
  // PNG, so the sizes are assembled into an ICO container by hand.
  const pngs = [];
  for (const size of SIZES) {
    const img = await render(win, size);
    pngs.push({ size, buffer: img.toPNG() });
  }
  writeFileSync(path.join(root, "build", "icon.ico"), buildIco(pngs));

  console.log("Wrote assets/icon.png and build/icon.ico");
  app.quit();
});

/** Packs PNG buffers into an .ico container. */
function buildIco(entries) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // reserved
  header.writeUInt16LE(1, 2); // type: icon
  header.writeUInt16LE(entries.length, 4);

  const dir = Buffer.alloc(16 * entries.length);
  let offset = header.length + dir.length;
  entries.forEach((entry, i) => {
    const at = i * 16;
    dir.writeUInt8(entry.size >= 256 ? 0 : entry.size, at + 0);
    dir.writeUInt8(entry.size >= 256 ? 0 : entry.size, at + 1);
    dir.writeUInt8(0, at + 2); // palette
    dir.writeUInt8(0, at + 3); // reserved
    dir.writeUInt16LE(1, at + 4); // colour planes
    dir.writeUInt16LE(32, at + 6); // bits per pixel
    dir.writeUInt32LE(entry.buffer.length, at + 8);
    dir.writeUInt32LE(offset, at + 12);
    offset += entry.buffer.length;
  });

  return Buffer.concat([header, dir, ...entries.map((e) => e.buffer)]);
}
