"""CLI entrypoint.

Subsystems are imported lazily inside each command so that a missing optional
dependency only breaks the command that needs it — `--doctor` in particular
must keep working precisely when the environment is broken.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__, paths
from .logging_setup import setup_logging

log = logging.getLogger("jarvis.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m jarvis",
        description="JARVIS — bilingual (Hindi/English) voice assistant for Windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m jarvis                          start the assistant (voice + HUD API)
  python -m jarvis --text "chrome kholo"    run one command, no microphone
  python -m jarvis --text "shutdown" --dry-run    show what it would do
  python -m jarvis --doctor                 check the environment
  python -m jarvis --test-tts "namaste"     verify the Hindi voice
""",
    )
    p.add_argument("--version", action="version", version=f"jarvis {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--quiet", action="store_true", help="log to file only")

    run = p.add_argument_group("running")
    run.add_argument("--no-audio", action="store_true",
                     help="skip the microphone; HUD and --text still work")
    run.add_argument("--no-server", action="store_true",
                     help="skip the HUD API server")

    one = p.add_argument_group("one-shot")
    one.add_argument("--text", metavar="COMMAND",
                     help="run one typed command through the full pipeline and exit")
    one.add_argument("--dry-run", action="store_true",
                     help="with --text: resolve the action but don't perform it")

    diag = p.add_argument_group("diagnostics")
    diag.add_argument("--doctor", action="store_true",
                      help="check the environment and report what is and isn't ready")
    diag.add_argument("--list-devices", action="store_true",
                      help="list audio input/output devices with their indices")
    diag.add_argument("--list-skills", action="store_true",
                      help="list every skill, grouped by category")
    diag.add_argument("--list-voices", metavar="LANG", nargs="?", const="",
                      help="list Edge TTS voices, optionally filtered (e.g. hi)")
    diag.add_argument("--test-tts", metavar="TEXT",
                      help="speak some text and exit")
    diag.add_argument("--test-stt", metavar="AUDIO_FILE",
                      help="transcribe an audio file and print the result")
    diag.add_argument("--test-mic", metavar="SECONDS", nargs="?", const="5", type=str,
                      help="record from the microphone and transcribe it")
    diag.add_argument("--tune-wake-word", metavar="TIMES", nargs="?", const="5",
                      type=str,
                      help="measure the wake word on your voice and recommend a "
                           "threshold (say it TIMES times, default 5)")
    diag.add_argument("--apply", action="store_true",
                      help="with --tune-wake-word, write the recommendation to "
                           "config.yaml instead of only printing it")

    return p


# --- diagnostics -----------------------------------------------------------


def cmd_list_devices() -> int:
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        print(f"sounddevice unavailable: {exc}")
        print("Install dependencies first:  pip install -r requirements.txt")
        return 1

    default_in, default_out = sd.default.device
    print(f"{'idx':>4}  {'in':>3} {'out':>3}  {'rate':>6}  name")
    print("-" * 70)
    for idx, dev in enumerate(sd.query_devices()):
        marks = []
        if idx == default_in:
            marks.append("DEFAULT-IN")
        if idx == default_out:
            marks.append("DEFAULT-OUT")
        suffix = ("  <- " + ", ".join(marks)) if marks else ""
        print(
            f"{idx:>4}  {dev['max_input_channels']:>3} {dev['max_output_channels']:>3}  "
            f"{int(dev['default_samplerate']):>6}  {dev['name']}{suffix}"
        )
    print("\nSet audio.input_device in config.yaml to an index or a name substring.")
    return 0


def cmd_list_skills() -> int:
    from .skills import load_all, registry

    load_all()
    total = 0
    for category, specs in sorted(registry.by_category().items()):
        print(f"\n{category.upper()}")
        for spec in sorted(specs, key=lambda s: s.name):
            total += 1
            gate = {"safe": "  ", "confirm": " ?", "critical": " !"}[spec.risk.value]
            print(f" {gate} {spec.name:<22} {spec.description[:60]}")
            if spec.examples:
                print(f"      e.g. {spec.examples[0]!r}, {spec.examples[-1]!r}")
    print(f"\n{total} skills.   ? = asks to confirm   ! = always confirms\n")
    return 0


def cmd_list_voices(prefix: str) -> int:
    import asyncio

    from .tts.edge import list_voices

    try:
        voices = asyncio.run(list_voices(prefix))
    except Exception as exc:  # noqa: BLE001
        print(f"Could not fetch the voice list: {exc}")
        return 1

    for voice in voices:
        print(f"  {voice['ShortName']:<34} {voice['Gender']:<7} {voice['Locale']}")
    print(f"\n{len(voices)} voices. Set tts.voice_hi / tts.voice_en in config.yaml.")
    return 0


def cmd_test_tts(text: str) -> int:
    from .nlu.normalize import has_devanagari
    from .tts import create_speaker

    speaker = create_speaker()
    language = "hi" if has_devanagari(text) else "en"
    print(f"Speaking ({speaker.name}, {language}): {text!r}")
    ok = speaker.speak(text, language)
    speaker.close()
    print("Done." if ok else "Playback was interrupted or failed.")
    return 0 if ok else 1


def cmd_test_stt(path: str) -> int:
    from .audio.decode import decode_audio
    from .stt import create_transcriber

    try:
        audio, rate = decode_audio(path, target_rate=16000)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read {path}: {exc}")
        return 1

    print(f"Decoded {audio.shape[0] / rate:.1f}s at {rate} Hz. Transcribing...")
    transcript = create_transcriber().transcribe(audio, rate)
    print(f"\n  language : {transcript.language} ({transcript.language_probability:.2f})")
    print(f"  latency  : {transcript.latency:.2f}s")
    print(f"  text     : {transcript.text!r}\n")
    return 0


def cmd_test_mic(seconds: float) -> int:
    import numpy as np
    import sounddevice as sd

    from .audio.capture import resolve_device
    from .config import settings
    from .stt import create_transcriber

    from .audio.enhance import is_too_quiet, speech_rms

    device = resolve_device(settings.audio.input_device, "input")
    print(f"Recording {seconds:.0f}s — speak now...")
    recording = sd.rec(int(seconds * 16000), samplerate=16000, channels=1,
                       dtype="float32", device=device)
    sd.wait()
    audio = np.asarray(recording).ravel()

    level = speech_rms(audio, 16000)
    print(f"Captured {audio.shape[0] / 16000:.1f}s, peak {np.abs(audio).max():.3f}, "
          f"speech level {20 * np.log10(max(level, 1e-6)):.0f} dBFS")

    if is_too_quiet(audio, 16000):
        print("\n  That is too quiet for reliable recognition. Raise the input\n"
              "  level in Windows sound settings, move closer, or pick a\n"
              "  different microphone (see --list-devices).\n")

    transcript = create_transcriber().transcribe(audio, 16000)
    print(f"\n  language : {transcript.language} ({transcript.language_probability:.2f})")
    print(f"  certainty: {transcript.confidence:.2f}   (below 0.62 triggers a re-decode)")
    print(f"  backend  : {transcript.backend}")
    print(f"  text     : {transcript.text!r}\n")
    return 0


def cmd_tune_wake_word(times: int, apply: bool) -> int:
    """Measure the wake word against this microphone and this voice.

    Reported rather than silently applied unless asked: the threshold is a
    trade-off between repeating yourself and being woken by the television, and
    which side of it you want is not something a measurement can decide.
    """
    from .audio.capture import AudioCapture
    from .audio.wake_tune import explain, measure
    from .audio.wakeword import OpenWakeWord
    from .config import settings

    try:
        detector = OpenWakeWord(
            settings.wake_word.model, threshold=settings.wake_word.threshold,
            cooldown_sec=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Wake word unavailable: {type(exc).__name__}: {exc}")
        return 1

    capture = AudioCapture(device=settings.audio.input_device)
    print(f"\nSay \"hey jarvis\" {times} times, with a breath between each.")
    print("Say it the way you normally would — as two connected words, not")
    print("\"hey ... jarvis\". The model scores a pause between them far lower,")
    print("which is why saying it more slowly after being missed makes it worse.\n")

    last = [-1]

    def progress(stage: str, elapsed: float, score: float, counted: int) -> None:
        if stage == "quiet":
            print(f"\n  listening to the room... {elapsed:4.1f}s", end="", flush=True)
            return
        if counted != last[0]:
            last[0] = counted
            print(f"\n  heard {counted} of {times}...            ", end="", flush=True)

    try:
        capture.start()
        result = measure(detector, capture, say_times=times, on_progress=progress)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not open the microphone: {type(exc).__name__}: {exc}")
        return 1
    finally:
        capture.stop()

    print("\n" + " " * 46)
    print("=" * 66)
    for line in explain(result, settings.wake_word.threshold):
        print(line)
    print("=" * 66)

    recommended = result.recommended
    if recommended is None:
        return 1
    if not apply:
        print(f"\n  To use it:  set wake_word.threshold to {recommended:.2f} in "
              "config.yaml,")
        print("              or re-run this with --apply\n")
        return 0

    if _write_threshold(recommended):
        print(f"\n  wake_word.threshold set to {recommended:.2f}. "
              "Restart for it to take effect.\n")
        return 0
    return 1


def _write_threshold(value: float) -> bool:
    """Update `wake_word.threshold` in config.yaml, creating it if need be.

    Edited as text rather than dumped from the parsed tree, because a round trip
    through the YAML loader would throw away every comment in the file — and the
    comments are most of what makes that file usable.
    """
    import re

    from . import paths

    path = paths.CONFIG_FILE
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Written by --tune-wake-word.\n"
                "wake_word:\n"
                f"  threshold: {value:.2f}\n",
                encoding="utf-8",
            )
            print(f"  wrote {path}")
            return True

        text = path.read_text(encoding="utf-8")
        pattern = re.compile(r"^(\s*threshold\s*:\s*)([0-9.]+)", re.MULTILINE)
        if "wake_word:" in text and pattern.search(text):
            text = pattern.sub(lambda m: f"{m.group(1)}{value:.2f}", text, count=1)
        else:
            text = text.rstrip("\n") + f"\n\nwake_word:\nthreshold: {value:.2f}\n"
        path.write_text(text, encoding="utf-8")
        print(f"  updated {path}")
        return True
    except OSError as exc:
        print(f"  could not write {path}: {exc}")
        return False


def cmd_doctor() -> int:
    """Report readiness of every subsystem. Never raises."""
    from .config import settings

    ok, warn, fail = "  OK  ", " WARN ", " FAIL "
    problems = 0

    def line(status: str, label: str, detail: str = "") -> None:
        print(f"[{status}] {label:<26} {detail}")

    print(f"\nJARVIS {__version__} — environment check\n" + "=" * 72)

    print(f"\nPlatform {sys.platform}")
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 11):
        line(fail, "python version", "3.11+ required")
        problems += 1

    print("\nPaths")
    line(ok, "engine dir", str(paths.ENGINE_DIR))
    line(ok, "data dir", str(paths.DATA_DIR))
    line(ok if paths.CONFIG_FILE.exists() else warn, "config.yaml",
         str(paths.CONFIG_FILE) if paths.CONFIG_FILE.exists() else "using defaults")
    line(ok if paths.ENV_FILE.exists() else warn, ".env",
         "found" if paths.ENV_FILE.exists() else "copy .env.example to .env")

    # Required everywhere, then the ones that only exist on this platform.
    # Reporting `win32api` as missing-and-fatal on a Mac would make --doctor
    # impossible to pass there — and `scripts/build-engine` gates the whole
    # build on its exit code, so that would have blocked every macOS build.
    print("\nDependencies")
    required = [
        ("sounddevice", "microphone capture"),
        ("numpy", "audio buffers"),
        ("onnxruntime", "wake word + VAD inference"),
        ("openwakeword", "wake word models"),
        ("faster_whisper", "speech to text"),
        ("edge_tts", "Hindi/English neural speech"),
        ("av", "audio decoding"),
        ("rapidfuzz", "bilingual fuzzy matching"),
        ("indic_transliteration", "Devanagari handling"),
        ("anthropic", "LLM brain"),
        ("psutil", "device status"),
        ("fastapi", "HUD API"),
    ]
    if sys.platform == "win32":
        required += [
            ("win32api", "Windows control (pywin32)"),
            ("comtypes", "audio session control"),
            ("pycaw", "volume control"),
        ]
    elif sys.platform == "darwin":
        required += [("Quartz", "macOS window control (pyobjc)")]

    for module, purpose in required:
        try:
            __import__(module)
            line(ok, module, purpose)
        except Exception as exc:  # noqa: BLE001
            line(fail, module, f"{purpose} - {type(exc).__name__}: {exc}")
            problems += 1

    print("\nAudio")
    try:
        import sounddevice as sd

        from .audio.capture import resolve_device

        inputs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if inputs:
            idx = resolve_device(settings.audio.input_device, "input")
            chosen = sd.query_devices(idx if idx is not None else sd.default.device[0])
            line(ok, "input devices", f"{len(inputs)} found")
            line(ok, "configured mic", chosen["name"])
        else:
            line(fail, "input devices", "no microphone detected")
            problems += 1
    except Exception as exc:  # noqa: BLE001
        line(fail, "audio subsystem", str(exc))
        problems += 1

    print("\nModels")
    whisper_dir = paths.MODELS_DIR / "whisper"
    line(ok if whisper_dir.exists() else warn, "whisper",
         f"cached ({settings.stt.model})" if whisper_dir.exists()
         else f"will download {settings.stt.model} (~150 MB) on first run")
    if settings.stt.escalate and settings.stt.accurate_model:
        line(ok, "whisper (accurate pass)",
             f"{settings.stt.accurate_model}, used below "
             f"{settings.stt.min_confidence:.2f} confidence")
    else:
        line(warn, "whisper (accurate pass)",
             "off — unclear speech won't get a second opinion")
    silero = paths.MODELS_DIR / "silero_vad.onnx"
    line(ok if silero.exists() else warn, "silero vad",
         "cached" if silero.exists() else "will download (~2 MB) on first run")
    try:
        import openwakeword

        oww = __import__("pathlib").Path(openwakeword.__file__).parent / "resources" / "models"
        found = list(oww.glob("*.onnx")) if oww.exists() else []
        line(ok if found else warn, "wake word",
             f"{len(found)} model(s) cached" if found else "will download on first run")
    except Exception:  # noqa: BLE001
        line(warn, "wake word", "openwakeword not importable")

    print("\nSkills")
    try:
        from .skills import load_all, registry

        count = load_all()
        by_risk: dict[str, int] = {}
        for spec in registry.all():
            by_risk[spec.risk.value] = by_risk.get(spec.risk.value, 0) + 1
        line(ok, "registered", f"{count} skills "
             f"({by_risk.get('safe', 0)} safe, {by_risk.get('confirm', 0)} confirm, "
             f"{by_risk.get('critical', 0)} critical)")
    except Exception as exc:  # noqa: BLE001
        line(fail, "skill registry", str(exc))
        problems += 1

    print("\nPermissions")
    try:
        from .permissions import Capability, consent
        from .security import session

        snapshot = consent.snapshot()
        if not snapshot["asked"]:
            line(warn, "capability consent",
                 "not recorded yet — everything is allowed (command-line mode)")
        else:
            granted = [c for c in Capability if snapshot["granted"][c.value]]
            denied = [c for c in Capability if not snapshot["granted"][c.value]]
            line(ok, "capability consent", f"{len(granted)} of {len(Capability)} granted")
            if denied:
                line(warn, "  turned off", ", ".join(c.value for c in denied))
        line(ok if not settings.security.session_required else
             (ok if session.active else warn), "sign-in required",
             "yes" if settings.security.session_required else "no (command-line mode)")
    except Exception as exc:  # noqa: BLE001
        line(fail, "permissions", str(exc))
        problems += 1

    print("\nLanguage model")
    try:
        from . import llm

        snapshot = llm.selection.snapshot()
        source = "from .env" if snapshot["from_environment"] else "set in the app"
        line(ok if snapshot["has_key"] else warn, "api key",
             f"{source} — {snapshot['label']}, model {snapshot['model']}"
             if snapshot["has_key"]
             else "none yet. Rules still work; conversation and Q&A need a key")
        line(ok, "providers", ", ".join(sorted(llm.PROVIDERS)))
    except Exception as exc:  # noqa: BLE001
        line(fail, "provider registry", str(exc))
        problems += 1

    try:
        from .stt.selection import selection as speech

        snapshot = speech.snapshot()
        line(ok if snapshot["has_key"] else warn, "cloud speech",
             f"{snapshot['label']} — used only when the local model comes back empty"
             if snapshot["has_key"] else "off — local speech only")
    except Exception as exc:  # noqa: BLE001
        line(fail, "speech selection", str(exc))
        problems += 1

    print("\n" + "=" * 72)
    if problems:
        print(f"{problems} problem(s) found. Run:  pip install -r requirements.txt\n")
        return 1
    print("All required components present.\n")
    return 0


# --- running ---------------------------------------------------------------


def cmd_text(text: str, dry_run: bool) -> int:
    from .orchestrator import orchestrator

    orchestrator.start(with_audio=False)
    reply = orchestrator.handle_text(text, source="text", dry_run=dry_run)
    print(f"\n  {reply}\n" if reply else "\n  (no reply)\n")
    orchestrator.stop()
    return 0


def cmd_run(no_audio: bool, no_server: bool) -> int:
    import signal

    from .orchestrator import orchestrator

    orchestrator.start(with_audio=not no_audio)

    def shutdown(*_: object) -> None:
        print()
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    try:
        signal.signal(signal.SIGTERM, shutdown)
    except (AttributeError, ValueError):
        pass

    if no_server:
        print("\nListening. Press Ctrl+C to stop.\n")
        try:
            import threading

            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            orchestrator.stop()
        return 0

    from .server import serve

    try:
        serve(orchestrator)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO, quiet=args.quiet)
    paths.ensure_dirs()

    if args.list_devices:
        return cmd_list_devices()
    if args.doctor:
        return cmd_doctor()
    if args.list_skills:
        return cmd_list_skills()
    if args.list_voices is not None:
        return cmd_list_voices(args.list_voices)
    if args.test_tts:
        return cmd_test_tts(args.test_tts)
    if args.test_stt:
        return cmd_test_stt(args.test_stt)
    if args.test_mic:
        return cmd_test_mic(float(args.test_mic))
    if args.tune_wake_word:
        return cmd_tune_wake_word(int(args.tune_wake_word), args.apply)
    if args.text:
        return cmd_text(args.text, args.dry_run)

    return cmd_run(args.no_audio, args.no_server)


if __name__ == "__main__":
    sys.exit(main())
