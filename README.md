# Local Timelapse Tray Recorder

Tray-based local timelapse recorder with:
- System tray app (click tray icon menu to show mini dashboard)
- Mini dashboard with `Start`, `Pause/Resume`, `Stop`
- Editable speed factor before start
- Save-As popup on stop (choose where to save)
- Pause state visible in video (dim + pause sign)
- Dashboard-style elapsed timer and progress bar overlaid into video

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (Python)

```powershell
python timelapse_recorder.py
```

After launch:
- App starts in system tray
- Use tray menu `Show Dashboard` to open mini dashboard
- Set speed factor, then click `Start`

## Build Executable (.exe)

```powershell
pip install pyinstaller
pyinstaller --noconfirm --onefile --noconsole --name TimelapseTray timelapse_recorder.py
```

Output executable:
- `dist\TimelapseTray.exe`

## Useful Options

```powershell
python timelapse_recorder.py --input-fps 24 --speed-factor 10 --output-fps 24 --format mp4 --max-width 1280
```

- `--input-fps`: capture fps (default `24`)
- `--speed-factor`: default dashboard speed factor (editable before recording)
- `--output-fps`: saved video fps (default `24`)
- `--format`: `mp4` or `avi`
- `--monitor`: monitor index (`1` is usually primary)
- `--max-width`: downscale width (`0` disables downscale)
- `--pause-dim-alpha`: pause dim strength (`0..1`)
