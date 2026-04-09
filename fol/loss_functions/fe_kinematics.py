
import jax.numpy as jnp

def b_matrix_2d(DN_DX:jnp.array) -> jnp.array:
    """
    Construct the 2D strain-displacement (B) matrix.

    The returned matrix maps nodal displacement degrees of freedom to the
    2D small-strain vector in Voigt form (ε_xx, ε_yy, γ_xy).

    Args:
        DN_DX:
            Derivatives of shape functions with respect to spatial coordinates.
            Expected shape is (n_nodes, 2), where columns correspond to
            dN/dx and dN/dy.

    Returns:
        jax.numpy.ndarray:
            The 2D B-matrix with shape (3, 2 * n_nodes).
    """
    B = jnp.zeros((3, 2 * DN_DX.shape[0]))
    indices = jnp.arange(DN_DX.shape[0])
    B = B.at[0, 2 * indices].set(DN_DX[indices,0])
    B = B.at[1, 2 * indices + 1].set(DN_DX[indices,1])
    B = B.at[2, 2 * indices].set(DN_DX[indices,1])
    B = B.at[2, 2 * indices + 1].set(DN_DX[indices,0])  
    return B

def b_matrix_3d(DN_DX:jnp.array) -> jnp.array:
    """
    Construct the 3D strain-displacement (B) matrix.

    The returned matrix maps nodal displacement degrees of freedom to the
    3D small-strain vector in Voigt form
    (ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz).

    Args:
        DN_DX:
            Derivatives of shape functions with respect to spatial coordinates.
            Expected shape is (n_nodes, 3), where columns correspond to
            dN/dx, dN/dy, and dN/dz.

    Returns:
        jax.numpy.ndarray:
            The 3D B-matrix with shape (6, 3 * n_nodes).
    """
    B = jnp.zeros((6,3*DN_DX.shape[0]))
    index = jnp.arange(DN_DX.shape[0]) * 3
    B = B.at[0, index + 0].set(DN_DX[:,0])
    B = B.at[1, index + 1].set(DN_DX[:,1])
    B = B.at[2, index + 2].set(DN_DX[:,2])
    B = B.at[3, index + 0].set(DN_DX[:,1])
    B = B.at[3, index + 1].set(DN_DX[:,0])
    B = B.at[4, index + 1].set(DN_DX[:,2])
    B = B.at[4, index + 2].set(DN_DX[:,1])
    B = B.at[5, index + 0].set(DN_DX[:,2])
    B = B.at[5, index + 2].set(DN_DX[:,0])
    return B

def d_matrix_2d(young_modulus:float,poisson_ratio:float) -> jnp.array:
    """
    Construct the 2D constitutive (D) matrix for isotropic linear elasticity
    under plane stress assumptions.

    Args:
        young_modulus:
            Young's modulus E of the material.
        poisson_ratio:
            Poisson's ratio ν of the material.

    Returns:
        jax.numpy.ndarray:
            The 2D constitutive matrix with shape (3, 3) relating the
            plane-stress strain vector (ε_xx, ε_yy, γ_xy) to the stress vector
            (σ_xx, σ_yy, τ_xy).
    """
    return jnp.array([[1,poisson_ratio,0],[poisson_ratio,1,0],[0,0,(1-poisson_ratio)/2]]) * (young_modulus/(1-poisson_ratio**2))

def d_matrix_3d(young_modulus:float,poisson_ratio:float) -> jnp.array:
    """
    Construct the 3D constitutive (D) matrix for isotropic linear elasticity.

    This matrix relates the 3D small-strain vector in Voigt form
    (ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz) to the corresponding stress vector
    (σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz).

    Args:
        young_modulus:
            Young's modulus E of the material.
        poisson_ratio:
            Poisson's ratio ν of the material.

    Returns:
        jax.numpy.ndarray:
            The 3D constitutive matrix with shape (6, 6).

    Raises:
        ValueError:
            If the provided parameters lead to a singular constitutive law,
            e.g., ν approaching 0.5 (near-incompressible limit) or invalid
            values that cause division by zero.
    """
    # construction of the constitutive matrix
    c1 = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    c2 = c1 * (1.0 - poisson_ratio)
    c3 = c1 * poisson_ratio
    c4 = c1 * 0.5 * (1.0 - 2.0 * poisson_ratio)
    D = jnp.zeros((6,6))
    D = D.at[0,0].set(c2)
    D = D.at[0,1].set(c3)
    D = D.at[0,2].set(c3)
    D = D.at[1,0].set(c3)
    D = D.at[1,1].set(c2)
    D = D.at[1,2].set(c3)
    D = D.at[2,0].set(c3)
    D = D.at[2,1].set(c3)
    D = D.at[2,2].set(c2)
    D = D.at[3,3].set(c4)
    D = D.at[4,4].set(c4)
    D = D.at[5,5].set(c4)
    return D

def n_matrix_2d(N_vec:jnp.array) -> jnp.array:
    """
    Construct the 2D shape function (N) matrix.

    The returned matrix maps nodal displacement degrees of freedom to the
    interpolated displacement vector at a point in the element.

    Args:
        N_vec:
            Shape function values evaluated at a point (e.g., Gauss point).
            Expected shape is (n_nodes,).

    Returns:
        jax.numpy.ndarray:
            The 2D N-matrix with shape (2, 2 * n_nodes).
    """
    N_mat = jnp.zeros((2, 2 * N_vec.size))
    indices = jnp.arange(N_vec.size)   
    N_mat = N_mat.at[0, 2 * indices].set(N_vec)
    N_mat = N_mat.at[1, 2 * indices + 1].set(N_vec)    
    return N_mat

def n_matrix_3d(N_vec:jnp.array) -> jnp.array:
    """
    Construct the 3D shape function (N) matrix.

    The returned matrix maps nodal displacement degrees of freedom to the
    interpolated displacement vector at a point in the element.

    Args:
        N_vec:
            Shape function values evaluated at a point (e.g., Gauss point).
            Expected shape is (n_nodes,).

    Returns:
        jax.numpy.ndarray:
            The 3D N-matrix with shape (3, 3 * n_nodes).
    """
    N_mat = jnp.zeros((3,3*N_vec.size))
    N_mat = N_mat.at[0,0::3].set(N_vec)
    N_mat = N_mat.at[1,1::3].set(N_vec)
    N_mat = N_mat.at[2,2::3].set(N_vec)
    return N_mat