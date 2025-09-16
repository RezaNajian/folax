# -------------------------------------------------------------------------
# Hyper_elastic_2d.py  –  linear Hooke (plane strain) and compressible Neo‑Hookean
# -------------------------------------------------------------------------
# Authors: Jerry Paul Varghese
# License: FOL/LICENSE
# 13-Sep-2025
# -------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from functools import partial

from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.loss_functions.fe_loss import FiniteElementLoss

# ------------------%%%%%%%-----------------------
# HELPERS for compressible Neo‑Hookean (2‑D)
# ------------------%%%%%%%-----------------------

def _F_flat_2D(F):
    return F.reshape(-1)            #Flattens a 2×2 deformation gradient F → shape (4,).


def _build_BF_2D(DN_DX, nnode):     #Builds a 4 × (2∙nnode) matrix BF such that F_flat = I_flat + BF @ q_e.
    '''Return 4×ndof matrix such that  F_flat = I_flat + BF @ q_e.'''
    BF = jnp.zeros((4, 2 * nnode))   #This is the geometric part of the updated small-strain formulation used for Neo-Hookean.
    a = jnp.arange(nnode)
    for i in range(2):  # i = 0,1                       
        for J in range(2):
            row = 2 * i + J  # 0 …. 3
            col = 2 * a + i
            BF = BF.at[row, col].set(DN_DX[a, J])
    return BF

#_make_neo_hookean_2D
#Returns a closure get_P_C_psi(F) that, for a given F:
#computes 1st-Piola stress P (via automatic grad of the strain-energy density psi)
#• returns the 4×4 material tangent C = ∂P/∂F and
#• returns the scalar energy psi(F).
#Everything is JIT-compiled and statically callable.

def _make_neo_hookean_2D(mu: float, kappa: float):
    """Return P, C, psi for 2D (plane-strain-type) compressible Neo-Hookean."""
    def psi(F):
        J  = jnp.linalg.det(F)
        Js = jnp.maximum(J, 1e-3)   # avoid log(0) or negative
        C  = F.T @ F
        I1 = jnp.trace(C)
        return (mu/2.0)*(I1 - 2.0) - mu*jnp.log(Js)  + (kappa/2.0)*(jnp.log(Js))**2

    P_fn   = jax.grad(psi)                      # 2x2 -> 2x2
    P_flat = lambda F: _F_flat_2D(P_fn(F))      # -> (4,)
    dPdF   = jax.jacfwd(P_flat)                 # -> (4,4)

    @partial(jax.jit, static_argnums=())
    def get_P_C_psi(F):
        P = P_fn(F)
        C = dPdF(F).reshape(4, 4)
        return P, C, psi(F)

    return get_P_C_psi


# --------------%%%%%%%%-----------------------------------
# Unified mechanical loss (2‑D)
# --------------%%%%%%%%-----------------------------------
# "linear": isotropic Hooke, plane-strain (D matrix built in Voigt form).
#"neo_hookean": compressible Neo-Hookean with mu, kappa.
# Sets body-force vector (defaults to 0 for NH).

class Neohookean2D(FiniteElementLoss):
    '''2‑D loss: linear (plane‑strain) and Neo‑Hookean (plane‑strain).'''

    def Initialize(self):
        super().Initialize()
        if 'material_dict' not in self.loss_settings:
            fol_error('material_dict must be provided in loss_settings')
        mat = self.loss_settings['material_dict']
        self.model = mat.get('model', 'linear').lower()
        self.dim = 2

        if self.model == 'linear':
            E, nu = mat['young_modulus'], mat['poisson_ratio']
            #self.D = self._hooke_matrix_plane_stress(E, nu)    # plane-stress (just for reference)
            self.D = self._hooke_matrix_plane_strain(E, nu)     # plane-strain
            print(f'OK Initialized 2-D linear elasticity: E={E}, nu={nu}')
            self._get_P_C_psi = None   # not used in linear case

            self.body_force = jnp.array(mat.get('body_force', jnp.zeros((self.dim, 1))))
        elif self.model == 'neo_hookean':
            self.mu, self.kappa = float(mat['mu']), float(mat['kappa'])
            print(f'OK--> Initialized 2-D Neo‑Hookean: mu={self.mu}, kappa={self.kappa}')
            self._get_P_C_psi = _make_neo_hookean_2D(self.mu, self.kappa)

            self.body_force = jnp.zeros((self.dim, 1))
        else:
            fol_error(f'Unknown material model {self.model}')

    # --------------------------------------------------------------
    # Hooke matrix (plane‑stress) -onöy used for reference here
    # --------------------------------------------------------------
    def _hooke_matrix_plane_stress(self, E, nu):
        c = E / (1 - nu ** 2)
        return c * jnp.array([[1, nu, 0],
                              [nu, 1, 0],
                              [0, 0, (1 - nu) / 2]])
    

    # -------------------<<<<<-------------------------------------------
    # Hooke matrix (plane‑strain)
    # -------------------<<<<<-------------------------------------------

    def _hooke_matrix_plane_strain(self, E, nu):
        """Isotropic linear elasticity, plane strain (Voigt: [εxx, εyy, εxy])."""
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * jnp.array([
            [1.0 - nu,   nu,         0.0],
            [nu,         1.0 - nu,   0.0],
            [0.0,        0.0,        0.5 * (1.0 - 2.0 * nu)],
        ])  


    # ------------------------<<<<<<--------------------------------------
    # Shape‑function matrices
    # ------------------------<<<<<<--------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _Bmat2D(self, DN_DX):
        n = DN_DX.shape[0]
        i = jnp.arange(n)
        B = jnp.zeros((3, 2 * n))
        B = B.at[0, 2 * i].set(DN_DX[:, 0])
        B = B.at[1, 2 * i + 1].set(DN_DX[:, 1])
        B = B.at[2, 2 * i].set(DN_DX[:, 1])
        B = B.at[2, 2 * i + 1].set(DN_DX[:, 0])
        return B

    @partial(jax.jit, static_argnums=(0,))
    def _Nmat2D(self, N):
        i = jnp.arange(N.size)
        Nmat = jnp.zeros((2, 2 * N.size))
        Nmat = Nmat.at[0, 2 * i].set(N)
        Nmat = Nmat.at[1, 2 * i + 1].set(N)
        return Nmat

    # --------------------->>>>>>>>-----------------------------------------
    # Element routine
    # --------------------->>>>>>>>-----------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def ComputeElement(self, X_e, dummy_controls, u_e):
        nnode = X_e.shape[0]
        gp_pts, gp_wts = self.fe_element.GetIntegrationData()

        def gp_loop(xi, w):
            DN_DX0 = self.fe_element.ShapeFunctionsGlobalGradients(X_e, xi)
            Jmat = self.fe_element.Jacobian(X_e, xi)
            detJ = jnp.linalg.det(Jmat)

            if self.model == 'linear':
                B = self._Bmat2D(DN_DX0)
                Nvec = self.fe_element.ShapeFunctionsValues(xi)
                Nmat = self._Nmat2D(Nvec)
                Ke_gp = w * detJ * (B.T @ self.D @ B)
                f_gp = w * detJ * (Nmat.T @ self.body_force)
                re_gp = Ke_gp @ u_e.reshape(-1, 1) - f_gp
                energy_gp = 0.5 * (u_e.reshape(-1, 1).T @ (Ke_gp @ u_e.reshape(-1, 1)))[0, 0] - (u_e.reshape(-1, 1).T @ f_gp)[0, 0]
            else:
                # inside gp_loop, NH branch
                u_mat  = u_e.reshape((nnode, 2))
                BF     = _build_BF_2D(DN_DX0, nnode)      # (4 x 2n)
                F      = jnp.eye(2) + u_mat.T @ DN_DX0    # (2 x 2)
                P, C, psi_val = self._get_P_C_psi(F)
                Pvec   = _F_flat_2D(P).reshape(-1, 1)
                Ke_gp  = w * detJ * (BF.T @ C @ BF)
                re_gp  = w * detJ * (BF.T @ Pvec)
                energy_gp = w * detJ * psi_val            # true strain energy


            return Ke_gp, re_gp, energy_gp

        Ke_gps, re_gps, en_gps = jax.vmap(gp_loop)(gp_pts, gp_wts)
        Se = jnp.sum(Ke_gps, axis=0)
        re = jnp.sum(re_gps, axis=0)
        energy = jnp.sum(en_gps)
        return energy, re, Se

    # ------------------------->>>>>>>>-------------------------------------
    # Total loss
    # ------------------------->>>>>>>>-------------------------------------
    def ComputeTotalLoss(self, control_vars, u_pred):
        total_energy = 0.0
        for elem_id in self.fe_mesh.GetElementIds():
            X_e = self.fe_mesh.GetElementCoordinates(elem_id)
            u_e = self.fe_mesh.ExtractElementNodalVector(u_pred, elem_id)
            dummy_controls = None
            energy_e, _, _ = self.ComputeElement(X_e, dummy_controls, u_e)
            total_energy += energy_e
        return total_energy

    # --------------------------<<<<<<<<------------------------------------
    # __call__
    # --------------------------<<<<<<<<-----------------------------------
    def __call__(self, control_batch, u_pred_batch):
        def loss_for_sample(control_vars, u_pred):
            return self.ComputeTotalLoss(control_vars, u_pred)
        batch_losses = jax.vmap(loss_for_sample)(control_batch, u_pred_batch)
        return jnp.mean(batch_losses)

# ---------------------%%%%%%----------------------------
# Convenience subclasses
# ---------------------%%%%%%----------------------------

class Neohookean2DTri(Neohookean2D):
    def __init__(self, name, loss_settings, fe_mesh: Mesh):
        super().__init__(name, {**loss_settings,
                                'compute_dims': 2,
                                'ordered_dofs': ['Ux', 'Uy'],
                                'element_type': 'triangle'}, fe_mesh)


class Neohookean2DQuad(Neohookean2D):
    def __init__(self, name, loss_settings, fe_mesh: Mesh):
        loss_settings.setdefault('num_gp', 2)
        super().__init__(name, {**loss_settings,
                                'compute_dims': 2,
                                'ordered_dofs': ['Ux', 'Uy'],
                                'element_type': 'quad'}, fe_mesh)
