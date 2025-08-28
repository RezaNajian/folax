import pytest
import os
import numpy as np
import pyvista as pv
from fol.tools.plotter import Plotter3D

@pytest.fixture
def tmp_vtk(tmp_path):
    mesh = pv.Cube().triangulate()
    pts = mesh.points.shape[0]
    # Sample 0: non-zero error
    mesh.point_data['U_FOL_0'] = np.zeros((pts, 3))
    mesh.point_data['U_FE_0']  = np.ones((pts, 3))
    # Sample 1: zero error
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


def test_find_best_and_derived_fields(tmp_vtk):
    p = Plotter3D(vtk_path=tmp_vtk, warp_factor=1.0, config={})
    p.find_best_sample()
    # Should pick sample '1' which has zero error
    assert p.best_id == '1'
    assert p.fields == {
        'K_field': 'K_1',
        'U_FOL':   'U_FOL_1',
        'U_FE':    'U_FE_1'
    }
    # Provide K_1 and compute derived fields
    p.mesh['K_1'] = np.full(p.mesh.n_points, 0.5)
    p.compute_derived_fields()
    expected_norm = np.linalg.norm([2,2,2])
    np.testing.assert_allclose(p.mesh['U_FOL_mag'], expected_norm)
    np.testing.assert_allclose(p.mesh['U_FE_mag'], expected_norm)
    np.testing.assert_allclose(p.mesh['abs_error'], 0.0)


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
    # Monkey-patch screenshot to accept any args
    def fake_screenshot(*args, **kwargs):
        # last positional arg is filename
        calls.append(args[-1])
    monkeypatch.setattr(pv.Plotter, 'screenshot', fake_screenshot)
    fname = "out.png"
    p.render_panel(p.mesh, field=None, clim=None, title="T", fname=fname, show_edges=False)
    expected = os.path.join(os.path.dirname(tmp_vtk), fname)
    assert calls == [expected]
