import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch


def _make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def export_to_json(results: dict, filepath: str | Path) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    serializable = _make_serializable(results)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return filepath


def export_to_csv(results: dict, filepath: str | Path) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    flat = {}
    for k, v in results.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            arr = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            if arr.ndim <= 2:
                flat[k] = arr.flatten().tolist()
        elif isinstance(v, list) and all(isinstance(i, (int, float, str)) for i in v):
            flat[k] = v
        elif isinstance(v, (int, float, str)):
            flat[k] = [v]

    max_len = max((len(v) for v in flat.values()), default=0)
    for k in flat:
        if len(flat[k]) < max_len:
            flat[k] = flat[k] + [None] * (max_len - len(flat[k]))

    df = pd.DataFrame(flat)
    df.to_csv(filepath, index=False)
    return filepath


def export_to_html(results: dict, figures: list, filepath: str | Path) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>TinyInterp Export</title></head><body>",
        "<h1>TinyInterp Analysis Results</h1>",
    ]

    html_parts.append("<h2>Parameters</h2><table border='1' cellpadding='4'>")
    for k, v in results.items():
        if isinstance(v, (str, int, float, bool)):
            html_parts.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
    html_parts.append("</table>")

    for i, fig in enumerate(figures):
        html_parts.append(f"<h2>Figure {i + 1}</h2>")
        html_parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

    html_parts.append("</body></html>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return filepath


def get_download_button(results: dict, fmt: str, filename: str) -> None:
    if fmt == "json":
        serializable = _make_serializable(results)
        data = json.dumps(serializable, indent=2, ensure_ascii=False).encode("utf-8")
        mime = "application/json"
    elif fmt == "csv":
        buf = io.StringIO()
        flat = {}
        for k, v in results.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                arr = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                if arr.ndim <= 2:
                    flat[k] = arr.flatten().tolist()
            elif isinstance(v, list) and all(isinstance(i, (int, float, str)) for i in v):
                flat[k] = v
        if flat:
            max_len = max(len(v) for v in flat.values())
            for k in flat:
                if len(flat[k]) < max_len:
                    flat[k] = flat[k] + [None] * (max_len - len(flat[k]))
            pd.DataFrame(flat).to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8")
        mime = "text/csv"
    else:
        return

    st.download_button(
        label=f"Download {fmt.upper()}",
        data=data,
        file_name=filename,
        mime=mime,
    )
