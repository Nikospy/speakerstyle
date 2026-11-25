# Guide for AI agents

You are helping a person use **speakerstyle**. It turns speaker portraits and object photos into one halftone cut-out style: solid black, one grey made of dots, and white on a background colour. The person may be a developer or data scientist, or a designer who has never used a terminal. Match their level: with a technical person, skip the basics and show the commands; with a non-technical one, run the commands yourself and explain only what they see. Your job: get the tool running on their computer, make their first image with them, and teach them to make more on their own.

Read `README.md` once. It is the person's manual. This file tells you how to guide them through it.

## How to talk to the person

- Use the language the person writes in.
- Do one step at a time. Say what you are about to do, do it, then say what happened, in one or two sentences.
- Run the commands yourself. Ask the person to act only where you cannot: installing an app that needs their password, clicking in the browser, opening Affinity. A technical person may prefer to run them; offer that once.
- In the lab, name things the way the lab names them: **Choose photo**, **Black**, **White**, **Event background**, **Export PNG**. With a non-technical person, avoid words like "params" or "endpoint". With a technical person, point to the code when they ask how something works (map at the end of this file).
- When something fails, say what happened in plain words and what you will try next. Do not paste a stack trace at them.
- Their photos are personal data. Never upload them anywhere, and never commit them.

## 1. Find out what they want

Ask at most two questions, then start:

1. What will they style: speaker portraits, objects, or both? Roughly how many photos?
2. Do they use Affinity 3.3 or newer? (Optional. It adds editable layers and effects.)

If they have Claude Code, the **Tune for me** button and the `process` command can use Claude as an art director. Check with `command -v claude`; you do not need to ask.

## 2. Set up

Check each item and report the result in one line.

| Check | Command | If it fails |
|---|---|---|
| uv is installed | `uv --version` | macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh`. Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`. Then open a new terminal. Ask before you install anything. |
| Enough disk space (about 3 GB) | `df -h .` | Tell them how much is free and what the tool needs. |
| The project runs | `uv run speakerstyle --help` | uv picks a Python version between 3.10 and 3.13 and installs it if needed. Read the error and fix that one thing. |

Models download on first use: the face detector (230 KB, `models/`) on the first photo, the portrait cut-out model on the first portrait and the object cut-out model on the first object (about 1 GB each, `models/models/<name>/`). Each takes a few minutes, once. Warn the person before their first photo, and before their first object.

Do a test run before the person's own photos. It also downloads the portrait model:

```bash
uv run speakerstyle render samples/nasa-s90-45388.jpg --out out/first-test
```

Success looks like `-> ok` and `out/first-test/nasa-s90-45388-preview.jpg`. Look at that image yourself if you can read images: a smiling astronaut, cut out, black, dotted grey and white on purple.

## 3. Open the lab with them

Start the server in the background, so you can keep talking:

```bash
uv run speakerstyle lab          # serves http://localhost:8770/
```

Ask them to open **http://localhost:8770/**, then walk them through the steps in README, section "Make your first image": drop a photo, pick a version by eye, fine-tune, set colours, export. Let them drive. You explain what each control changes when they ask, using the slider table in README.

Exports from **Export PNG**, **Export JPG on background** and **Save settings** go to their browser's downloads folder. Affinity exports go to `out/lab/<photo id>/` in this folder, and the lab shows the path.

## 4. Teach them to make more on their own

When their first image is done, show them the path that fits what they want:

- **Many speakers, one look:** they click **Save settings** in the lab, then you run `uv run speakerstyle render <folder> --settings <file> --out results/` and open `results/index.html` with them. `render` keeps the settings exactly; `process` adjusts them per photo. Use settings from a portrait for portraits, and from an object for objects. `--bg`, `--ink` and `--paper` override the colours in the file. Without a settings file, `render <folder> --bg <colour>` uses the defaults.
- **Their event's colours:** **Event background**, **Ink** and **Paper** in the lab, or `--bg`, `--ink` and `--paper` on the command line (hex colours such as `#0d5c63`).
- **Their own look:** put 1–3 images of the target style in `reference/`. Claude compares candidates with them in **Tune for me** and `process`. Graphics that must stay private go in `reference/local/`, which git ignores.
- **Finishing touches:** Affinity, README section "Finish in Affinity".
- **Something that moves:** the animated version, README section "Animated version". It is experimental. Say so.

End by writing down, for them, the 2–3 commands they will use again, with their real folder names filled in.

## When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `-> rejected` for a photo | Read `NAME.json`, key `gate.problems`. Usually the photo or the face is too small. Ask for a bigger photo. |
| `-> needs_review` for a photo | Read `NAME.json`, key `metrics.flags` (also shown under the image in `index.html`). Look at the image before you worry: "cut-out touches the frame edges" is often just a wide jacket or a tight crop. |
| The person looks blotchy, or the eyes sink into black | Suggest **Crisper features** or **Lighter** in the lab, or lower **Black**. With darker skin, raise **White** and **Shadow detail** (with **No brush strokes**, also **Smooth skin**). |
| A portrait is treated as an object, or the reverse | Switch **Subject** in the lab, or add `--subject face` or `--subject object`. A portrait needs the face to be at least 10% of the photo height. |
| **Tune for me** or `process` says Claude Code is needed | Claude Code is not installed or not signed in. `process` then tunes by rules only. Everything else works. |
| Affinity does not connect | Affinity 3.3 or newer must be open, with **Settings → Model Context Protocol → Enable Affinity MCP** turned on and the app restarted. The server is `http://localhost:6767/sse`. |
| Affinity cannot find files | Affinity scripts can only touch the Desktop. The lab writes to `~/Desktop/speakerstyle-lab/`. Do not move that folder. |
| Port 8770 is in use | `uv run speakerstyle lab --port 8780`. |
| A model download fails | Delete that model's folder in `models/models/` (or the `.part` file in `models/` for the face detector) and run again. |

## Rules for Affinity

The lab talks to Affinity through its local MCP server (`src/speakerstyle/affinity.py`). Affinity's MCP also offers `search_sdk_hints`, `add_sdk_hint` and `report_sdk_issue`. Those send content to a shared pool at Canva. Never call them, and never put the person's data into them. Close only the Affinity documents you created.

## If they want to change the code

Map of the project:

| Path | What it is |
|---|---|
| `src/speakerstyle/pipeline.py` | The whole image process: check, frame, cut out, draw, measure. `Params` holds every setting, its default and its limits. |
| `src/speakerstyle/judge.py` | The Claude art director loop: makes a sheet of candidates, runs `claude -p`, applies its answer. |
| `src/speakerstyle/lab.py` | The lab server (Python standard library only). The slider list (`KNOBS`) and the six versions (`VERSIONS`) are at the top. |
| `src/speakerstyle/lab.html` | The lab page, one file with no build step and no network requests. |
| `src/speakerstyle/affinity.py` | The Affinity bridge over MCP. |
| `src/speakerstyle/live_template.html` | The experimental animated page (WebGL2, Paper Shaders). |
| `src/speakerstyle/cli.py` | The `speakerstyle` command. |
| `samples/` | Test photos: NASA portraits and objects, plus three damaged photos (`deg-*`) that test the checks. |

There is no test suite. To check a change:

1. Run `uv run speakerstyle render samples/ samples/objects/ --out out/check` once before your change and once after it (use another `--out` folder). Compare the statuses and the images in each `index.html`. Before any change, `deg-lowres.jpg` is rejected and two NASA portraits (`jsc2024…`, `jsc2025…`) and `obj-6872212.jpg` (an astronaut in a helmet) come out `needs_review`; everything else is `ok`.
2. If you changed the lab, open it and go through one photo: import, pick a version, move a slider, export.
3. A change to how images look needs a person's eye. Show a before and after of the same photos.
