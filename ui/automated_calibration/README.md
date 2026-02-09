# Automated calibration runner

This UI automates the manual workflow:

1. Connect to the Pi Zero 2 W access point (AP)
2. SSH into the Pi and start the EKF calibration webapp
3. After flying, pull logs to the laptop via `scp` (after clearing old local logs)
4. Delete remote logs on the Pi
5. Switch back to your internet hotspot
6. Open the Analysis window and chat with the model (it uses `python_sandbox` with numpy/pandas/matplotlib)

## Setup

1. Copy `run_config.example.json` to `run_config.json` in this folder.
2. Fill in:
   - `pi_ap.ssid` / `pi_ap.password`
   - `hotspot.ssid` / `hotspot.password`
   - `pi_ssh.user` and `pi_ssh.host` (e.g. `omrijsharon.local`)
   - `remote.webapp_cmd` and `remote.logs_dir`
   - `local.logs_dir` and `local.archive_root`
   - `analysis.*` (system prompt, debug mapping, optional code paths)

`run_config.json` is git-ignored because it contains Wi-Fi passwords.

## Run

From repo root:

`.\.venv\Scripts\python.exe scripts\automated_calibration_ui.py`

## Notes

- Windows Wi‑Fi connect is done via `netsh wlan ...` (best-effort).
- SSH/SCP require `ssh` and `scp` to be available on PATH (Windows OpenSSH is fine).
- The LLM analysis step uses `python_sandbox`, which runs in the current venv.
