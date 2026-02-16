# Dev Workspace (per-chat project)

The Chat UI can expose a per-chat “Dev Workspace” so the model can build and run code:
- Persistent project directory (HTML/JS/CSS/Python)
- Python venv per chat
- `pip install` into that venv (with **per-install user approval**)
- Optional static preview server (`python -m http.server`)

Workspace root:
- `chat_history/workspaces/<chat_id>/`

## Enable

Chat UI → Controls → Tools:
- Enable “Dev Workspace”
- (Optional) Allow “Dev Workspace pip installs”
- (Optional) Allow “Dev Workspace static HTTP server”

## Tools available to the model

- `workspace_info`
- `workspace_write`, `workspace_append`, `workspace_read`, `workspace_list`
- `workspace_create_venv`
- `workspace_pip_install` (prompts the user to approve)
- `workspace_run_python`
- `workspace_http_server_start`, `workspace_http_server_stop`

## Progress visibility

During `workspace_pip_install`, pip output is saved to:
- `chat_history/workspaces/<chat_id>/pip_install.log`

You can open that file while installation runs to see progress. The final tool output also includes the log path.

