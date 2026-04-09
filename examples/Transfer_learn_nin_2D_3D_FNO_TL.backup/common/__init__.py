# common package – shared utilities for all phase-scripts in mechanical_square/
#
# Convenience re-exports so callers can do:
#   from common import safe_rename, _parse_ids, ...   (general)
#   from common import _midz_slice_scalar, ...        (3D plotting)

from .script_utils import (
    safe_rename,
    _parse_ids,
    _sanitize_label,
    _clean_mkdir,
    _find_first_existing,
    _read_training_residual_csv,
    _summarize_training,
    _append_row_csv,
    _write_single_row_csv,
    _ensure_vtu_in_dir,
    _copy_training_artifacts,
    _copy_rms_plot_as_training_history,
    _patch_rng_keys,
)

from .plot_3d import (
    _midz_slice_scalar,
    _save_slice_png,
    _save_2x2_panel,
)

__all__ = [
    # script_utils
    "safe_rename",
    "_parse_ids",
    "_sanitize_label",
    "_clean_mkdir",
    "_find_first_existing",
    "_read_training_residual_csv",
    "_summarize_training",
    "_append_row_csv",
    "_write_single_row_csv",
    "_ensure_vtu_in_dir",
    "_copy_training_artifacts",
    "_copy_rms_plot_as_training_history",
    "_patch_rng_keys",
    # plot_3d
    "_midz_slice_scalar",
    "_save_slice_png",
    "_save_2x2_panel",
]
