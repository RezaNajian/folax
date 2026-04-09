"""
script_utils.py
~~~~~~~~~~~~~~~
Shared utility functions for all phase-scripts in mechanical_square/.

Includes:
  - Filesystem helpers : safe_rename, _clean_mkdir, _find_first_existing,
                         _ensure_vtu_in_dir
  - Sample-id helpers  : _parse_ids, _sanitize_label
  - CSV helpers        : _read_training_residual_csv, _summarize_training,
                         _append_row_csv, _write_single_row_csv
  - Artifact copy      : _copy_training_artifacts,
                         _copy_rms_plot_as_training_history
  - JAX/FNO compat     : _patch_rng_keys
"""

from __future__ import annotations

import csv
import glob
import os
import shutil

import numpy as np


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def safe_rename(src: str, dst: str) -> None:
    """Rename *src* → *dst* if *src* exists; silently overwrite *dst*."""
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)


def _clean_mkdir(path: str) -> None:
    """Remove *path* if it already exists, then recreate it as an empty dir."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _find_first_existing(patterns: list[str]) -> str | None:
    """Return the first existing file matched by any of the given glob patterns."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def _ensure_vtu_in_dir(export_dir: str) -> None:
    """
    If no *.vtu* file exists in *export_dir* but a *.vtk* does, convert the
    first .vtk to .vtu via meshio (optional dependency).
    """
    if sorted(glob.glob(os.path.join(export_dir, "*.vtu"))):
        return  # already have a .vtu

    vtks = sorted(glob.glob(os.path.join(export_dir, "*.vtk")))
    if not vtks:
        return

    try:
        import meshio  # type: ignore
    except Exception:
        print("[EXPORT] WARN: meshio not available, cannot convert .vtk -> .vtu")
        return

    src = vtks[0]
    dst = os.path.splitext(src)[0] + ".vtu"
    try:
        m = meshio.read(src)
        meshio.write(dst, m, file_format="vtu")
        print(f"[EXPORT] Converted {os.path.basename(src)} -> {os.path.basename(dst)}")
    except Exception as exc:
        print(f"[EXPORT] WARN: conversion failed: {exc}")


# ---------------------------------------------------------------------------
# Sample-id / label helpers
# ---------------------------------------------------------------------------

def _parse_ids(ids_str: str, n: int) -> list[int]:
    """
    Parse a sample-id specification string and return a de-duplicated list of
    integer indices.

    Supported formats (may be mixed with commas):
      ``"all"`` or ``"*"``  → all indices 0 … n-1
      ``"a-b"``             → inclusive range [a, b]
      ``"i"``               → single index i
    """
    s = str(ids_str).strip().lower()
    if s in ("all", "*"):
        return list(range(n))

    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a_s, b_s = tok.split("-", 1)
            a, b = int(a_s.strip()), int(b_s.strip())
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if not (0 <= i < n):
                    raise ValueError(f"sample id {i} out of range [0, {n - 1}]")
                out.append(i)
        else:
            i = int(tok)
            if not (0 <= i < n):
                raise ValueError(f"sample id {i} out of range [0, {n - 1}]")
            out.append(i)

    if not out:
        raise ValueError("ids_str parsed to empty list.")

    # de-duplicate, preserving order
    seen: set[int] = set()
    out2: list[int] = []
    for i in out:
        if i not in seen:
            out2.append(i)
            seen.add(i)
    return out2


def _sanitize_label(s: str) -> str:
    """Return a filesystem-safe lowercase label string."""
    s = str(s).strip().lower().replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))
    return s or "pkl"


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_training_residual_csv(csv_path: str):
    """
    Read a training-residual CSV produced by TrainingResidualTracker.

    Returns ``(epoch, resid, tloss)`` arrays, or ``None`` if the file cannot
    be read or does not contain a recognisable residual column.
    """
    if (csv_path is None) or (not os.path.exists(csv_path)):
        return None

    rows: list[dict] = []
    with open(csv_path, "r", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        return None

    cols = list(rows[0].keys())
    epoch_col = "epoch" if "epoch" in cols else cols[0]

    residual_col = None
    for candidate in ("residual_rms_batch_mean", "residual_rms", "residual"):
        if candidate in cols:
            residual_col = candidate
            break
    if residual_col is None:
        for c in cols:
            if "residual" in c.lower() and "rms" in c.lower():
                residual_col = c
                break
    if residual_col is None:
        return None

    total_loss_col = "total_loss" if "total_loss" in cols else None

    epoch = np.array([float(r[epoch_col]) for r in rows], dtype=float)
    resid = np.array([float(r[residual_col]) for r in rows], dtype=float)
    tloss = (
        np.array([float(r[total_loss_col]) for r in rows], dtype=float)
        if total_loss_col is not None
        else None
    )
    return epoch, resid, tloss


def _summarize_training(csv_path: str, residual_target: float) -> dict:
    """
    Return a summary dict with epoch / residual statistics from a training CSV.
    All values are ``np.nan`` when the file is missing or unreadable.
    """
    out = {
        "train_csv": csv_path,
        "epochs_completed": np.nan,
        "last_epoch": np.nan,
        "final_residual": np.nan,
        "min_residual": np.nan,
        "epoch_at_min": np.nan,
        "epoch_first_below_target": np.nan,
        "final_total_loss": np.nan,
    }
    data = _read_training_residual_csv(csv_path)
    if data is None:
        return out
    epoch, resid, tloss = data
    if len(epoch) == 0:
        return out

    out["epochs_completed"] = int(epoch[-1] + 1)
    out["last_epoch"] = int(epoch[-1])
    out["final_residual"] = float(resid[-1])
    out["min_residual"] = float(np.min(resid))
    out["epoch_at_min"] = float(epoch[int(np.argmin(resid))])

    idx = np.where(resid <= residual_target)[0]
    if len(idx) > 0:
        out["epoch_first_below_target"] = float(epoch[int(idx[0])])

    if tloss is not None and len(tloss) == len(epoch):
        out["final_total_loss"] = float(tloss[-1])

    return out


def _append_row_csv(csv_path: str, fieldnames: list[str], row: dict) -> None:
    """Append *row* to *csv_path*, writing a header row if the file is new."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_single_row_csv(csv_path: str, fieldnames: list[str], row: dict) -> None:
    """Write (overwrite) *csv_path* with a single data row."""
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Artifact copy helpers
# ---------------------------------------------------------------------------

def _copy_training_artifacts(train_case_dir: str, case_dir: str, train_tag: str) -> None:
    """
    Copy training artefacts (history image + residual CSV/PNG) from the
    shared parametric training folder into a per-sample folder so that every
    sample directory is self-contained.
    """
    # training history image
    for fn in ("training_history_parametric.png", "training_history.png"):
        src = os.path.join(train_case_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(case_dir, fn))
            break

    # residual CSV / PNG files
    patterns = [
        os.path.join(train_case_dir, f"{train_tag}*.csv"),
        os.path.join(train_case_dir, f"{train_tag}*.png"),
        os.path.join(train_case_dir, "*residual_rms*.csv"),
        os.path.join(train_case_dir, "*residual_rms*.png"),
    ]
    copied: set[str] = set()
    for pat in patterns:
        for src in sorted(glob.glob(pat)):
            base = os.path.basename(src)
            if base in copied:
                continue
            try:
                shutil.copy2(src, os.path.join(case_dir, base))
                copied.add(base)
            except Exception:
                pass


def _copy_rms_plot_as_training_history(case_dir: str, sample_id: int, tag: str) -> None:
    """
    Copy the tracker RMS plot to ``training_history_sample_<id>.png`` so that
    TL-OTF sample folders are consistent with OTF tooling expectations.
    """
    dst = os.path.join(case_dir, f"training_history_sample_{sample_id}.png")

    preferred = os.path.join(case_dir, f"{tag}_residual_rms.png")
    if os.path.exists(preferred):
        shutil.copyfile(preferred, dst)
        return

    candidates = sorted(glob.glob(os.path.join(case_dir, "*residual_rms*.png")))
    if candidates:
        shutil.copyfile(candidates[0], dst)


# ---------------------------------------------------------------------------
# JAX / FNO compatibility helpers
# ---------------------------------------------------------------------------

def _patch_rng_keys(fno_model):
    """
    Patch Flax NNX RNG key state for checkpoint-safety across JAX versions.

    Some versions of JAX store RNG keys as typed-key arrays; others expect raw
    uint32 data.  This function converts them to the raw form so that restored
    checkpoints are compatible regardless of JAX version.
    """
    from flax import nnx  # local import – avoids hard top-level JAX dependency
    import jax

    def _merge_state(dst: nnx.State, src: nnx.State) -> None:
        for k, v in src.items():
            if isinstance(v, nnx.State):
                _merge_state(dst[k], v)
            else:
                dst[k] = v  # type: ignore[index]

    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    _merge_state(state, rngs_key)
    return nnx.merge(graph_def, state)
