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

    device = resolve_device(settings.audio.input_device, "input")
    print(f"Recording {seconds:.0f}s — speak now...")
    recording = sd.rec(int(seconds * 16000), samplerate=16000, channels=1,
                       dtype="float32", device=device)
    sd.wait()
    audio = np.asarray(recording).ravel()
    print(f"Captured {audio.shape[0] / 16000:.1f}s, peak {np.abs(audio).max():.3f}")

    if np.abs(audio).max() < 0.01:
        print("\n  That's almost silent — check the microphone in config.yaml"
              " (see --list-devices).\n")

    transcript = create_transcriber().transcribe(audio, 16000)
    print(f"\n  language : {transcript.language} ({transcript.language_probability:.2f})")
    print(f"  text     : {transcript.text!r}\n")
    return 0


def cmd_doctor() -> int:
    """Report readiness of every subsystem. Never raises."""
    from .config import settings

    ok, warn, fail = "  OK  ", " WARN ", " FAIL "
    problems = 0

    def line(status: str, label: str, detail: str = "") -> None:
        print(f"[{status}] {label:<26} {detail}")

    print(f"\nJARVIS {__version__} — environment check\n" + "=" * 72)

    print(f"\nPython {sys.version.split()[0]} ({sys.executable})")
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

    print("\nDependencies")
    for module, purpose in [
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
        ("win32api", "Windows control (pywin32)"),
        ("comtypes", "audio session control"),
        ("fastapi", "HUD API"),
    ]:
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
         else f"will download {settings.stt.model} (~500 MB) on first run")
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

    print("\nSecrets")
    line(ok if settings.brain.api_key else warn, "ANTHROPIC_API_KEY",
         f"set (model {settings.brain.model})" if settings.brain.api_key
         else "missing - rules still work, no conversation/Q&A")
    line(ok if settings.stt.cloud_api_key else warn, "CLOUD_STT_API_KEY",
         "set" if settings.stt.cloud_api_key else "not set - local speech only")

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
    if args.text:
        return cmd_text(args.text, args.dry_run)

    return cmd_run(args.no_audio, args.no_server)


if __name__ == "__main__":
    sys.exit(main())
