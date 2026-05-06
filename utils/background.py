import threading
import traceback
from typing import Any, Callable

import streamlit as st

_TASKS: dict[str, dict] = {}


def submit_task(
    task_id: str,
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
) -> None:
    if kwargs is None:
        kwargs = {}

    if task_id in _TASKS and _TASKS[task_id]["state"]["status"] == "running":
        _TASKS[task_id]["state"]["status"] = "cancelled"

    lock = threading.Lock()
    state = {
        "status": "running",
        "progress": 0.0,
        "message": "Starting...",
        "result": None,
        "error": None,
    }

    def progress_callback(fraction: float, message: str) -> bool:
        with lock:
            if state["status"] == "cancelled":
                return False
            state["progress"] = min(max(fraction, 0.0), 1.0)
            state["message"] = message
            return True

    def wrapper():
        try:
            result = fn(progress_callback, *args, **kwargs)
            with lock:
                if state["status"] != "cancelled":
                    state["status"] = "done"
                    state["result"] = result
                    state["progress"] = 1.0
                    state["message"] = "Complete"
        except Exception:
            with lock:
                state["status"] = "error"
                state["error"] = traceback.format_exc()
                state["message"] = "Failed"

    thread = threading.Thread(target=wrapper, daemon=True)
    _TASKS[task_id] = {"thread": thread, "lock": lock, "state": state}
    thread.start()


def get_progress(task_id: str) -> dict | None:
    if task_id not in _TASKS:
        return None
    entry = _TASKS[task_id]
    with entry["lock"]:
        return dict(entry["state"])


def cancel_task(task_id: str) -> None:
    if task_id in _TASKS:
        entry = _TASKS[task_id]
        with entry["lock"]:
            if entry["state"]["status"] == "running":
                entry["state"]["status"] = "cancelled"


def is_task_running(task_id: str) -> bool:
    if task_id not in _TASKS:
        return False
    entry = _TASKS[task_id]
    with entry["lock"]:
        return entry["state"]["status"] == "running"


def render_progress_fragment(
    task_id: str,
    result_session_key: str,
    on_complete_message: str = "Analysis complete!",
):
    @st.fragment(run_every=2)
    def _poll():
        progress = get_progress(task_id)
        if progress is None:
            return

        status = progress["status"]

        if status == "running":
            st.progress(progress["progress"], text=progress["message"])
        elif status == "done":
            st.success(on_complete_message)
            st.session_state[result_session_key] = progress["result"]
            st.rerun()
        elif status == "error":
            st.error(f"Task failed:\n```\n{progress['error']}\n```")
        elif status == "cancelled":
            st.warning("Task was cancelled.")

    _poll()
