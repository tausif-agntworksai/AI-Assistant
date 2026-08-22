# Fixed bugs, and what stops them coming back

One row per bug that has been fixed. The **Guarded by** column is the point of
this file: if a row has no test, the bug is one refactor away from returning.

The rule is that a fix isn't finished until its symptom has a test named after
it. Tests are named for what the user experienced, not for the mechanism —
`test_opening_a_url_works_when_startfile_is_missing`, not
`test_shell_open_getattr`. When one fails, the failure should read like a bug
report.

---

## Cross-platform

All five were invisible on Windows and fatal on macOS. Guarded by
[`tests/test_cross_platform.py`](../engine/tests/test_cross_platform.py), which
fakes `sys.platform` and deletes the Windows-only attributes so a Windows test
run exercises the macOS path.

| Symptom | Root cause | Guarded by |
|---|---|---|
| Every website, file, folder, settings page and WhatsApp chat answers "That didn't work" | `winutil.shell_open` called `os.startfile`, which **does not exist** off Windows and raises `AttributeError` — not an `OSError`, so it escaped the handler. One line, ~15 of the 72 skills. | `test_opening_a_url_works_when_startfile_is_missing`, `test_opening_a_folder_uses_xdg_open_on_linux`, `test_windows_still_uses_startfile` |
| Assistant appears to work but never speaks, with nothing in the log | `SapiSpeaker` declared `available = True` and imported nothing until the first utterance, so `create_speaker` handed back a Windows-only speaker on any platform. It accepted every reply and played none. Now the constructor import-checks, so the factory falls through to `NullSpeaker`. | `test_tts_falls_back_to_silence_not_to_a_speaker_that_cannot_speak` |
| `pip install -r requirements.txt` fails outright on macOS | `pywin32`, `comtypes`, `pycaw` and `WMI` had no environment markers, and none of them publish non-Windows wheels — so nothing installed, including the ~30 skills that are already portable. | Markers in [`requirements.txt`](../engine/requirements.txt); no test (a packaging fact, not a code path) |
| `--doctor` can never pass on macOS, which blocks the build | The dependency list named `win32api` and `comtypes` as required unconditionally, and `cmd_doctor` returns 1 on any problem. `build-engine` gates on that exit code. | `test_the_doctor_can_pass_without_pywin32` |
| "system status" answers "That didn't work" | `psutil.disk_usage("C:\\")` — hardcoded, and wrong even on Windows for a profile on another drive. Now `Path.home().anchor`. | `test_the_disk_report_does_not_assume_a_c_drive` |
| 650 MB of speech models land in a bare `~/Jarvis` | `DATA_DIR` fell back to `Path.home() / "Jarvis"` whenever `LOCALAPPDATA` was unset, i.e. on every non-Windows machine. Now per-platform: `%LOCALAPPDATA%`, `~/Library/Application Support`, `$XDG_DATA_HOME`. | `test_the_data_directory_is_not_a_bare_folder_in_the_home_directory` |
| "open videos" finds nothing on a Mac | macOS calls the folder `Movies`. Also `recycle bin` mapped to `shell:RecycleBinFolder`, which is meaningless outside Explorer. | `test_the_video_folder_is_called_movies_on_macos` |
| App index silently globs the working directory | `Path(os.environ.get("APPDATA", ""))` is `Path(".")` when the variable is unset, so `rglob("*")` walked wherever the engine happened to be started from. | `test_the_start_menu_scan_does_not_walk_the_working_directory` |

## Hearing

| Symptom | Root cause | Guarded by |
|---|---|---|
| Saying "haan" to a confirmation cancels the action | `haan`, `ji` and `ok` are all on Whisper's list of phrases it invents out of silence, so the hallucination filter deleted every spoken *yes* — and the gate treats anything that isn't a clear yes as a refusal. Confirmations now transcribe with `short_answer=True`, which keeps them. | `test_a_spoken_yes_is_kept_when_a_yes_is_what_we_asked_for` in [`test_hearing.py`](../engine/tests/test_hearing.py) |
| "chrome kholo" said with a pause arrives as "chrome" and matches nothing | A fixed 700 ms endpoint ended the turn during a mid-sentence pause. The window is now adaptive: it stretches while barely any speech has happened. | `test_a_pause_mid_command_does_not_end_the_turn`, `test_a_settled_utterance_still_ends_promptly` in [`test_vad.py`](../engine/tests/test_vad.py) |
| Answering a Hindi question in English | Whisper's language label is close to a coin flip on a short romanised command — it labelled "Chrome kholo" Polish. The label is now one input among several, behind script detection and vocabulary. | [`test_language.py`](../engine/tests/test_language.py) |
| "explain quantum computing to me" answered in Hindi, right after a Hindi command | The conversation-inheritance rule — correct for a two-word follow-up like "aur?" — was applied to any utterance with no Hindi markers, however long. Inheritance is now limited to utterances of two words or fewer; a longer sentence with no Hindi evidence is English. | `test_a_whole_english_sentence_does_not_inherit_hindi` |
| A bare "volume" or "battery" mid-Hindi-conversation flipped the reply to English | Technology loanwords were listed as *English* markers, but Hindi speakers use them constantly inside Hindi sentences, so they are evidence of nothing. Removed from the marker list. | `test_a_technology_loanword_alone_does_not_flip_to_english` |
| A quiet microphone produces confident nonsense | Whisper's answer to a quiet clip is not "I'm not sure", it is the wrong words. Clips are now conditioned (rumble removal, loudness normalisation, padding) before recognition. | [`test_enhance.py`](../engine/tests/test_enhance.py) |
| Same command needs saying two or three times | A single-tier recogniser with no recovery: a mishearing routed to no skill and the assistant went silent, so the whole utterance had to be repeated from the wake word. Now the audio is re-decoded by a larger model, and a failure asks out loud and keeps listening. | [`test_hearing.py`](../engine/tests/test_hearing.py) |

## Security

| Symptom | Root cause | Guarded by |
|---|---|---|
| Any web page could shut the machine down | The engine's loopback API had `allow_origins=["*"]` and no authentication. Any page could `fetch("http://127.0.0.1:8756/command")`. Now a per-launch bearer token, and any real browser `Origin` is refused outright. | `test_a_web_page_origin_is_refused_even_with_the_token`, `test_every_other_route_needs_the_token` in [`test_server.py`](../engine/tests/test_server.py) |
| A revoked capability still ran | Consent was drawn in the UI but not enforced in the engine. Now checked in `registry.execute()` **before** the confirmation prompt. | `test_a_revoked_capability_blocks_the_skill` in [`test_permissions.py`](../engine/tests/test_permissions.py) |
| `python -m jarvis` refused every command | The session lock was enforced unconditionally, but a command-line launch has no app to sign in with. Now it mirrors the microphone's rule. | `test_a_command_line_launch_needs_no_sign_in` in [`test_server.py`](../engine/tests/test_server.py) |

## Startup and packaging

| Symptom | Root cause | Guarded by |
|---|---|---|
| "The engine did not respond in time" after a 15-minute wait, on any machine where port 8756 was taken | The desktop app scans for a free port and passes it in `JARVIS_PORT` — and **nothing in the engine read it**. The app polled the port it chose while the engine sat on the one from `config.yaml`. Nothing in the error message points at a port conflict. | `test_the_engine_binds_the_port_the_app_asked_for`, `test_a_nonsense_port_falls_back_to_the_configured_one` |
| A bare `ERR_FAILED` on launch, with no hint that a second copy was running | `app.quit()` is a request, not a return: without an explicit guard the rest of startup still ran, built a window and began loading a page that was then torn down underneath it. | The `isPrimaryInstance` guard in [`main.ts`](../desktop/src/main.ts) |
| The frozen build would start and then fail on the first question | The model providers are imported inside `_build_registry()`, so PyInstaller's static analysis never saw them — `No module named jarvis.llm.anthropic_provider` at runtime only. | `hiddenimports` in [`jarvis-engine.spec`](../engine/jarvis-engine.spec), verified by curling the frozen binary's `/llm` |
| `npm run dist` dies in the icon step | This shell exports `ELECTRON_RUN_AS_NODE=1`, so `electron scripts/make-icon.mjs` runs as plain Node and `require("electron")` has no `BrowserWindow`. | Documented under *Environment traps* below |

## Desktop app

| Symptom | Root cause | Guarded by |
|---|---|---|
| No way to minimise or quit from the sign-in or permission screens | The React rewrite gave the gate screens a draggable strip and no buttons. On a frameless window that meant someone who could not get past sign-in — no authenticator to hand, a forgotten password — had to use Task Manager. A gate you cannot back out of is a trap. | [`WindowControls.tsx`](../desktop/renderer-src/src/components/WindowControls.tsx), rendered on all three screens |
| Duplicate log lines in the HUD after a screen change | The four `on*` bridge methods returned nothing, so a React effect's cleanup had no disposer to call and every remount added another `ipcRenderer.on`. They now return an unsubscribe function. | Types in [`preload.ts`](../desktop/src/preload.ts); the React hooks that depend on it arrive with the GUI rewrite |
| Unhandled `ERR_ABORTED` rejection on launch | Sign-in state settles twice in quick succession, and two overlapping `loadFile` calls make Chromium abort the first. Navigations are now serialised through one promise chain. | `showScreen` in [`main.ts`](../desktop/src/main.ts) |
| A blank white window in the packaged build only | Vite's default `base` emits `<script src="/assets/...">`, and under `file://` a leading slash resolves to the filesystem root. Invisible in `vite dev` and in any app that loads over HTTP. Fixed with `base: "./"`. | The comment in [`vite.config.ts`](../desktop/vite.config.ts), plus checking `renderer-dist/index.html` contains `./assets/` after every build |
| The renderer's CSP forbade its own bundle | `script-src 'unsafe-inline'` permits inline scripts but not external ones, and a Vite build emits external modules. Now `script-src 'self'` — which is also strictly safer than what it replaced. | The CSP in each of the three entry HTML files |
| `npm install` leaves Vite unable to build | npm now withholds install scripts by default, so esbuild's postinstall never fetched its platform binary. | `allowScripts` in [`package.json`](../desktop/package.json), same convention as the sibling app |
| A packaged build could fail with no evidence anywhere | A packaged Windows app is a GUI-subsystem binary with no console, so `console.error` went nowhere — and the window is frameless with no menu, so DevTools could not be opened by hand either. A white screen left nothing to look at. Diagnostics now go to `desktop.log`, reachable from the tray, and a successful paint is logged too so its *absence* is the signal. | [`log.ts`](../desktop/src/log.ts) and the `did-finish-load` / `did-fail-load` / `render-process-gone` handlers in `main.ts` |
| "renderer died" logged on every clean quit | Shutting down kills the renderer and Chromium reports that as a crash, so the new handler cried wolf every time. What matters is a renderer that dies *before* painting — that is the white screen. | The `quitting || painted` guard in `main.ts` |
| Every launch loaded the same page twice | The startup call passed `force: true`, but restoring the session emits a change whose handler had usually already begun loading that screen. | The comment on the `showScreen(requiredScreen())` call in `main.ts`; visible as a single "painted" line in `desktop.log` |

---

## Environment traps

Not bugs in this code, but things that cost time twice.

- **`electron .` from a tool shell runs as plain Node.** This machine exports
  `ELECTRON_RUN_AS_NODE=1`, so `require("electron")` resolves to the npm
  wrapper — a path string — and the main process dies with
  `TypeError: Cannot read properties of undefined (reading 'isPackaged')`.
  Launch with `(unset ELECTRON_RUN_AS_NODE; ./node_modules/electron/dist/electron.exe .)`.
- **PyInstaller cannot cross-compile.** A macOS engine build has to happen on a
  Mac; there is no way to produce it from Windows.
- **A `file://` renderer needs `base: "./"` in Vite.** The default absolute base
  emits `/assets/...`, which resolves to the filesystem root and white-screens —
  and only in the packaged build, never in `vite dev`.
