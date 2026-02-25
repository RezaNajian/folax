"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations

from typing import List, Sequence, Union, Optional

import jax.numpy as jnp
from flax import nnx
from abc import ABC, abstractmethod

Number = Union[float, int]

# JAX/Flax NNX implementation of GridEmbeddingND.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/layers/embeddings.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/layers/embeddings.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.

def regular_grid_nd(
    resolutions: List[int], grid_boundaries: List[List[int]] = [[0, 1]] * 2
):
    """regular_grid_nd generates a tensor of coordinate points that
    describe a bounded regular grid.

    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid.

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding
    """
    assert len(resolutions) == len(
        grid_boundaries
    ), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start, stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(jnp.linspace(start, stop, res + 1)[:-1])

    return jnp.meshgrid(*meshgrid_inputs, indexing="ij")


class Embedding(nnx.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self):
        pass

class GridEmbeddingND(nnx.Module):
    """GridEmbeddingND applies a simple positional embedding as a regular ND grid.
    
    It expects inputs of shape (batch, d_1, ..., d_n, channels)

    Parameters
    ----------
    in_channels : int
        number of channels in input
    dim : int
        dimensions of positional encoding to apply
    grid_boundaries : list, optional
        coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """

    def __init__(
        self,
        in_channels: int,
        dim: int = 2,
        grid_boundaries: Optional[List[List[Number]]] = [[0, 1], [0, 1]],
    ):
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(
            grid_boundaries
        ), f"Error: expected grid_boundaries to be an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self) -> int:
        return self.in_channels + self.dim

    def grid(self, spatial_dims: Sequence[int], dtype:jnp.dtype) -> jnp.ndarray:
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : sizes of spatial resolution
        dtype : str
            dtype to encode data

        Returns
        -------
        jnp.ndarray
            output grids to concatenate
        """

        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(spatial_dims, grid_boundaries=self.grid_boundaries)
            grid = nnx.data(jnp.stack(grids_by_dim, axis=-1).astype(dtype))
            self._grid = grid
            self._res = spatial_dims

        return self._grid

    def __call__(self, data: jnp.ndarray, batched: bool = True) -> jnp.ndarray:
        """
        Params
        --------
        data: torch.Tensor
            assumes shape (batch (optional), x_1, x_2, ...x_n, channels)
        batched: bool
            whether data has a batch dim
        """

        if not batched and data.ndim == self.dim + 1:
            data = data[None, ...]  # add batch axis

        batch_size = data.shape[0]
        spatial_dims = data.shape[1:-1]
        dtype = data.dtype
        grids = self.grid(spatial_dims=data.shape[1:-1], dtype=dtype)
        grids = jnp.broadcast_to(grids, (batch_size, *spatial_dims, self.dim))
        out = jnp.concatenate([data, grids], axis=-1)
        return out
    
class SinusoidalEmbedding(Embedding):
    """
    Sinusoidal positional embedding for enriching coordinate inputs with spectral information.

    This class provides sinusoidal positional embeddings in two styles: Transformer-style
    and NeRF-style. It lifts low-dimensional coordinates into a richer spectral representation
    by encoding them as periodic functions (sines and cosines) at multiple frequencies.

    The embedding enhances a model's ability to capture fine-scale variations and high-frequency
    dynamics by providing a hierarchy of frequency components alongside the original coordinates.

    Parameters
    ----------
    in_channels : int
        Number of input channels to embed (dimensionality of input coordinates)
    num_freqs : int, optional
        Number of frequency levels L in the embedding. Each level contributes
        a sine and cosine pair, resulting in 2L output channels per input channel.
        By default, set to the number of input channels.
    embedding_type : {'transformer', 'nerf'}, optional
        Type of embedding to apply, by default 'transformer'

        Transformer-style [1]_:
        For each input coordinate p and frequency level k (0 ≤ k < L):
        - g(p)_{2k} = sin(p / max_positions^{k/L})
        - g(p)_{2k+1} = cos(p / max_positions^{k/L})

        NeRF-style [2]_:
        For each input coordinate p and frequency level k (0 ≤ k < L):
        - g(p)_{2k} = sin(2^k * π * p)
        - g(p)_{2k+1} = cos(2^k * π * p)

    max_positions : int, optional
        Maximum number of positions for transformer-style encoding, by default 10000.
        Only used when embedding_type='transformer'.


    Notes
    -----
    - Input shape: (batch, n_in, in_channels) or (n_in, in_channels)
    - Output shape: (batch, n_in, 2*num_freqs*in_channels) or (n_in, 2*num_freqs*in_channels)
    - Ensure the highest frequency satisfies the Nyquist criterion:
      - Transformer: f_max < N/2 where N is the number of sampling points
      - NeRF: 2^{L-1} < N/2, i.e., L < 1 + log₂(N/2)


    Examples
    --------
    See `examples/layers/plot_sinusoidal_embeddings.py` for comprehensive visualizations


    References
    ----------
    .. [1] Vaswani, A. et al. "Attention Is All You Need".
           NeurIPS 2017, https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

    .. [2] Mildenhall, B. et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis".
           ArXiv 2020, https://arxiv.org/pdf/2003.08934
    """

    def __init__(
        self,
        in_channels: int,
        num_frequencies: Optional[int] = None,
        embedding_type: str = "nerf",
        max_positions: int = 10000,
        modulation: Optional[str] = "frequency",  # None | "amplitude" | "frequency"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.modulation = modulation
        self._grid = None
        self._res = None

        # verify embedding type
        allowed_embeddings = ["nerf", "transformer"]
        assert (
            embedding_type in allowed_embeddings
        ), f"Error: embedding_type expected one of {allowed_embeddings}, received {embedding_type}"
        self.embedding_type = embedding_type
        if self.embedding_type == "transformer":
            assert (
                max_positions is not None
            ), "Error: max_positions must have an int value for \
                transformer embedding."
        self.max_positions = max_positions

        allowed_modulations = [None, "amplitude", "frequency"]
        assert modulation in allowed_modulations, (
            f"modulation must be one of {allowed_modulations}, got {modulation}"
        )

    def grid(self, spatial_dims: Sequence[int], dtype:jnp.dtype) -> jnp.ndarray:
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : sizes of spatial resolution
        dtype : str
            dtype to encode data

        Returns
        -------
        jnp.ndarray
            output grids to concatenate
        """

        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(spatial_dims, grid_boundaries=[[0, 1], [0, 1]])
            grid = nnx.data(jnp.stack(grids_by_dim, axis=-1).astype(dtype))
            self._grid = grid
            self._res = spatial_dims

        return self._grid
        
    @property
    def out_channels(self):
        """
        required property for linking/composing model layers
        """
        return 2 * self.num_frequencies * self.in_channels

    def __call__(self, x, m: Optional[jnp.ndarray] = None):
        """
        Parameters
        -----------
        x: jnp.ndarray
            Shape (grid_1, ..., grid_n, in_channels) or 
                    (batch, grid_1, ..., grid_n, in_channels)

        m : jnp.ndarray or None
            Constant parameter to encode.
            Shape (), (batch,), or (batch, 1)

        """
        assert x.ndim in [3,4], f"Error: expected inputs of shape (batch, grid_1, ..., grid_n, {x.shape[-1]})\
            or (grid_1, ..., grid_n, channels), got inputs with ndim={x.ndim}, shape={x.shape}"
        if x.ndim == 3:
            batched = False
            x = x[None,:]
        else:
            batched = True
        batch_size = x.shape[0]
        spatial_dims = x.shape[1:-1]    
        dtype = x.dtype
        n_in = len(spatial_dims)

        grids = self.grid(spatial_dims=spatial_dims, dtype=dtype)
        grids = jnp.broadcast_to(grids, (batch_size, *spatial_dims, n_in))

        # ----------------------------
        # Frequencies
        # ----------------------------
        if self.embedding_type == "nerf":
            freqs = 2 ** jnp.arange(0, self.num_frequencies) * jnp.pi

        elif self.embedding_type == "transformer":
            k = jnp.arange(0, self.num_frequencies) / self.num_frequencies * 2
            freqs = (1 / self.max_positions) ** k

        # ----------------------------
        # Parameter handling
        # ----------------------------
        if self.modulation is not None:
            assert m is not None, "Parameter m must be provided for modulation"

            m = jnp.asarray(m, dtype=dtype)
            if m.ndim == 0:
                m = jnp.broadcast_to(m, (batch_size,))
            elif m.ndim == 1:
                assert m.shape[0] == batch_size, (
                    f"Expected m with batch dimension {batch_size}, got {m.shape}"
                )
            elif m.ndim == 2 and m.shape[1] == 1:
                assert m.shape[0] == batch_size, (
                    f"Expected m with batch dimension {batch_size}, got {m.shape}"
                )
                m = m[:, 0]
            else:
                raise ValueError(
                    "m must have shape (), (batch,), or (batch, 1)"
                )

            m = m.reshape((batch_size,) + (1,) * (n_in + 2))        # to (b, n, m, c, L)

        # ----------------------------
        # Frequency modulation: g(m * x)
        # ----------------------------
        if self.modulation == "frequency":
            grids = grids * m

        # Outer product: (b, n, m, c, L)
        angles = jnp.einsum("bnmc,l->bnmcl", grids, freqs)

        sin = jnp.sin(angles)
        cos = jnp.cos(angles)

        # ----------------------------
        # Amplitude modulation: m * g(x)
        # ----------------------------
        if self.modulation == "amplitude":
            sin = sin * m
            cos = cos * m

        # outer product of wavenumbers and position coordinates
        # shape b, n_in * channels, len(freqs)
        # -------------------------------------------------
        # Outer product: positions × frequencies
        # x:     (b, n, m, c)
        # freqs: (L,)
        # →      (b, n, m, c, L)
        # -------------------------------------------------
        
        # embed coordinates and sinusoidal layers
        emb = jnp.stack((sin, cos), axis=-1)  # (b, n, m, c, L, 2)
        emb = emb.reshape(batch_size, *spatial_dims, -1)

        if not batched:
            x = x[0]
            emb = emb[0]

        # emb_m = jnp.broadcast_to(emb, grids.shape[:-1] + (emb.shape[-1],))
        out = jnp.concatenate([x, emb], axis=-1)
        
        return out
