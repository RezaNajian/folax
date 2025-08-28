import pytest
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from fol.tools.plotter import Plotter3D


# -----------------------
# synthetic VTK
# -----------------------

@pytest.fixture
def tmp_vtk(tmp_path):
    mesh = pv.Cube().triangulate()
    pts = mesh.points.shape[0]

    # Sample 0: non-zero magnitude-difference error
    mesh.point_data['U_FOL_0'] = np.zeros((pts, 3))
    mesh.point_data['U_FE_0']  = np.ones((pts, 3))

    # Sample 1: zero magnitude-difference error (best should be '1')
    mesh.point_data['U_FOL_1'] = 2 * np.ones((pts, 3))
    mesh.point_data['U_FE_1']  = 2 * np.ones((pts, 3))

    vtk_file = tmp_path / "mesh.vtk"
    mesh.save(str(vtk_file))
    return str(vtk_file)


@pytest.fixture
def tmp_vtk_single(tmp_path):
    mesh = pv.Cube().triangulate()
    pts = mesh.points.shape[0]
    mesh.point_data['U_FOL_0'] = np.random.rand(pts, 3)
    mesh.point_data['U_FE_0']  = np.random.rand(pts, 3)
    vtk_file = tmp_path / "mesh_single.vtk"
    mesh.save(str(vtk_file))
    return str(vtk_file)


# -----------------------
# Unit tests
# -----------------------

def test_find_best_and_derived_fields(tmp_vtk):
    p = Plotter3D(vtk_path=tmp_vtk, warp_factor=1.0, config={})
    p.find_best_sample()

    # Should pick sample '1' which has zero |‖U_FOL‖ − ‖U_FE‖|
    assert p.best_id == '1'
    assert p.fields == {
        'K_field': 'K_1',
        'U_FOL':   'U_FOL_1',
        'U_FE':    'U_FE_1'
    }

    # Provide K_1 and compute derived fields (magnitudes + abs_error)
    p.mesh['K_1'] = np.full(p.mesh.n_points, 0.5)
    p.compute_derived_fields()

    expected_norm = np.linalg.norm([2, 2, 2])
    np.testing.assert_allclose(p.mesh['U_FOL_mag'], expected_norm)
    np.testing.assert_allclose(p.mesh['U_FE_mag'], expected_norm)
    np.testing.assert_allclose(p.mesh['abs_error'], 0.0)  # |‖U_FOL‖ − ‖U_FE‖|


def test_find_best_single_sample(tmp_vtk_single):
    p = Plotter3D(vtk_path=tmp_vtk_single, warp_factor=1.0, config={})
    p.find_best_sample()
    # Only sample '0' exists
    assert p.best_id == '0'
    assert p.fields['U_FOL'] == 'U_FOL_0'
    assert p.fields['U_FE']  == 'U_FE_0'


def test_apply_cut_reduces_bounds(tmp_vtk):
    p = Plotter3D(vtk_path=tmp_vtk, warp_factor=1.0, config={})
    orig_bounds = p.mesh.bounds
    cut_bounds = p.apply_cut(p.mesh).bounds
    assert cut_bounds[1] <= orig_bounds[1]
    assert cut_bounds[3] <= orig_bounds[3]
    assert cut_bounds[5] <= orig_bounds[5]


def test_render_panel_invokes_screenshot(tmp_vtk, monkeypatch):
    p = Plotter3D(vtk_path=tmp_vtk, warp_factor=1.0, config={})
    calls = []

    # Monkey-patch screenshot to record the saved filename
    def fake_screenshot(self, filename, *args, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(pv.Plotter, 'screenshot', fake_screenshot)

    fname = "out.png"
    p.render_panel(p.mesh, field=None, clim=None, title="T", fname=fname, show_edges=False)
    expected = os.path.join(os.path.dirname(tmp_vtk), fname)
    assert calls == [expected]


# -----------------------
# Integration-ish checks
# -----------------------

def test_render_all_panels_uses_abs_error_and_respects_fixed_error_clim(tmp_vtk, monkeypatch):
    """
    Ensure the stitched overview pipeline uses the same abs_error field
    (|‖U_FOL‖ − ‖U_FE‖|) and honors fixed_error_clim for the error panel.
    """
    cfg = {
        "output_image": "overview.png",
        "fixed_K_clim": [0.0, 1.0],
        "fixed_error_clim": [0.0, 0.123],  # assert this gets used
        "warp_factor_overview": 1.0,
    }
    p = Plotter3D(vtk_path=tmp_vtk, warp_factor=1.0, config=cfg)
    p.find_best_sample()

    # Provide required elasticity field for the chosen best sample
    p.mesh[p.fields['K_field']] = np.full(p.mesh.n_points, 0.5)

    # Capture render_panel calls (field, clim, fname)
    calls = []
    def fake_render_panel(self, mesh_obj, field, clim, title, fname, show_edges=None):
        calls.append({"field": field, "clim": clim, "fname": fname, "title": title})

    # Avoid heavy sub-pipelines and file IO for contour/slices and diagonal plot
    monkeypatch.setattr(Plotter3D, 'render_panel', fake_render_panel)
    monkeypatch.setattr(Plotter3D, 'render_contour_slice', lambda self: None)
    monkeypatch.setattr(Plotter3D, 'render_diagonal_plot', lambda self: None)
    monkeypatch.setattr(plt, 'imread', lambda path: np.zeros((10, 10, 3), dtype=np.uint8))  # stitch step

    p.render_all_panels()

    # Find the error panel call (panel7.png by convention)
    err_calls = [c for c in calls if c["fname"] == "panel7.png"]
    assert len(err_calls) == 1
    err_call = err_calls[0]

    # Should use 'abs_error' (|‖U_FOL‖ − ‖U_FE‖|)
    assert err_call["field"] == "abs_error"

    # Should honor fixed_error_clim
    assert err_call["clim"] == cfg["fixed_error_clim"]
