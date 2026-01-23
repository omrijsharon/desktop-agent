# Short-term plan (controller + project alignment)

This file tracks the immediate cleanup tasks identified after reviewing `project.md` and `controller.py`.

## Controller (`controller.py`) fixes / improvements

- [x] **Fix the misleading usage text in the docstring**
  - Update the “Usage (manual testing)” section to reference the actual file (`controller.py`) and show correct examples for PowerShell.
  - Rationale: reduce confusion for anyone trying to test the controller quickly.

- [x] **Make ESC emergency stop match the documented behavior**
  - Decide one of these approaches and implement it consistently:
    1) **Hard-exit** the process after releasing inputs (e.g., `os._exit(code)`), or
    2) Keep the controller as a library/server and **only guarantee `release_all()`**, while surfacing a stop signal to the main loop.
  - Rationale: currently ESC triggers `SystemExit` in a background thread which does **not reliably terminate the whole process**.

- [x] **Rename or rework `os_exit()` to be truthful and reliable**
  - If hard-exiting: rename to something like `hard_exit()` and actually call `os._exit`.
  - If not hard-exiting: rename to `request_exit()` and implement a cooperative shutdown mechanism.
  - Rationale: the current name/comment suggests OS-level hard exit, but it raises `SystemExit`.

- [x] **Prevent step-mode approval from blocking shutdown indefinitely**
  - Replace `Event.wait()` with a loop that uses a timeout and checks a server stop flag.
  - Ensure `stop` can interrupt a pending approval wait.
  - Rationale: currently `_approved.wait()` can hang forever unless an `approve` message arrives (or ESC is pressed).

- [x] **Align step-mode protocol with `project.md`**
  - Either:
    - document the `{"op": "approve"}` control-plane op in `project.md`, or
    - remove `approve` from the controller and implement step-mode at the executor/planner layer.
  - Rationale: right now project documentation describes step-mode conceptually but doesn’t define the wire-level approval op.

- [x] **Tighten error responses for easier debugging**
  - Expand error response shape to include e.g. `op`, `exception_type`, maybe `traceback` behind a debug flag.
  - Rationale: `{ok:false, error:str(e)}` is workable but becomes painful when integrating multiple modules.

## Repo hygiene / dev setup

- [x] **Add a Python virtual environment and document it**
  - Create `.venv` in repo root.
  - Add `.venv/` to `.gitignore`.
  - Add a minimal `requirements.txt` (even if initially empty) as the project grows.
  - Rationale: avoids global Python dependency drift and keeps installs reproducible.

- [x] **Add basic repo metadata**
  - Add `README.md` with:
    - quickstart
    - how to run controller server
    - how to create/activate venv
  - Rationale: improves onboarding and reduces repeated setup questions.

## Tests

- [x] **Add automated tests for controller server behavior**
  - Add unit tests for `ControlsServer.handle()`:
    - `ping`, `approve`, `stop`, `release_all`
    - step-mode gating logic (approval required)
    - invalid payloads return structured errors
  - Add tests for `WindowsControls.release_all()` bookkeeping (without calling real SendInput).
  - Provide a small fake/stub controls implementation so CI can run on any machine without generating real input.
