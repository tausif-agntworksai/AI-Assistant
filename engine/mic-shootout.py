"""Records every candidate microphone at once and reports which one hears you.

Jarvis skips inputs that look virtual, so on a machine where the default mic is
something like DroidCam it can end up listening to a jack nobody talks into.
One utterance through all the devices simultaneously settles which is which.

    .venv\\Scripts\\python.exe mic-shootout.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import sounddevice as sd

from jarvis.audio.vad import create_vad

SECONDS = 6.0
RATE = 16000
PHRASE = "hey jarvis, chrome kholo"


def candidates():
    """Every distinct physical input, best guess at one entry per device."""
    seen, out = set(), []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"].replace("\n", " ").strip()
        key = name.lower()
        # Skip the aggregate/mapper entries and duplicate host-API copies.
        if any(s in key for s in ("sound mapper", "primary sound", "midi", "stereo mix")):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append((idx, name))
    return out


def main():
    devices = candidates()
    streams, buffers = [], {}

    for idx, name in devices:
        buf = []
        try:
            stream = sd.InputStream(
                device=idx, samplerate=RATE, channels=1, dtype="float32",
                blocksize=512,
                callback=(lambda b: lambda indata, frames, t, status: b.append(indata[:, 0].copy()))(buf),
            )
            stream.start()
            streams.append((idx, name, stream))
            buffers[idx] = buf
        except Exception as exc:
            print(f"  [skip] {idx:>3} {name[:44]:<44} {str(exc)[:36]}")

    print()
    print("=" * 72)
    print(f"  SPEAK NOW - say:  \"{PHRASE}\"")
    print(f"  Recording {SECONDS:.0f} seconds from {len(streams)} microphones...")
    print("=" * 72)
    for remaining in range(int(SECONDS), 0, -1):
        print(f"    {remaining}...", flush=True)
        time.sleep(1)

    for _, _, stream in streams:
        stream.stop()
        stream.close()

    vad = create_vad("silero")
    n = vad.frame_size

    rows = []
    for idx, name, _ in streams:
        chunks = buffers[idx]
        if not chunks:
            rows.append((idx, name, 0.0, 0.0, 0.0))
            continue
        audio = np.concatenate(chunks).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio)))
        probs = [vad.probability(audio[i:i + n]) for i in range(0, audio.size - n, n)]
        vad.reset()
        speech = float(np.mean(np.array(probs) > 0.5)) if probs else 0.0
        rows.append((idx, name, rms, peak, speech))

    rows.sort(key=lambda r: r[4], reverse=True)

    print(f"\n{'idx':>4}  {'device':<46} {'rms':>8} {'peak':>7} {'speech':>7}")
    print("-" * 80)
    for idx, name, rms, peak, speech in rows:
        flag = "  <== HEARS YOU" if speech > 0.15 else ""
        print(f"{idx:>4}  {name[:46]:<46} {rms:>8.5f} {peak:>7.4f} {speech:>6.0%}{flag}")

    best = rows[0] if rows else None
    print()
    if best and best[4] > 0.15:
        print(f"Winner: device {best[0]} - {best[1]}")
        print("Pin it in engine\\config.yaml under audio.input_device, e.g.")
        print(f"    input_device: {best[0]}")
    else:
        print("No device detected speech. Either nothing was said during the")
        print("countdown, or Windows is blocking mic access for desktop apps")
        print("(Settings > Privacy & security > Microphone).")


if __name__ == "__main__":
    main()
