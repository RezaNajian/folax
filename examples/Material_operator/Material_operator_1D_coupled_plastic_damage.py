"""
Optuna hyperparameter tuning for the Brepols (2017) local level coupled damage-plasticity 

Tuned hyperparameters (8):
    - modes        : Fourier modes per SpectralConv1D
    - width        : channel width (must be divisible by num_heads)
    - depth        : number of stacked FNO+attention blocks
    - lr           : AdamW learning rate
    - batch_size   : mini-batch size
    - weight_decay : AdamW weight decay
    - omega0       : SIREN activation frequency scaling
    - num_heads    : attention heads (paired with width via a joint suggestion)
"""

import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pickle
import time as time_module
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# ==================================
# 1. BREPOLS DAMAGE-PLASTICITY MODEL 
# ==================================
@jax.jit
def f_dam(D):
    return 1.0 - D


@jax.jit
def energy_release_rate_Y(ee, xi_p, D, prop):
    E, s0, e_p, Y0, r_d, s_d = prop
    fd = f_dam(D)
    return fd * (E * ee ** 2 + e_p * xi_p ** 2)


@jax.jit
def yield_function_Fp(sigma, xi_p, D, prop):
    E, s0, e_p, Y0, r_d, s_d = prop
    fd = f_dam(D)
    q_p = fd ** 2 * e_p * xi_p
    return jnp.abs(sigma) - fd * s0 - q_p


@jax.jit
def damage_function_Fd(ee, xi_p, xi_d, D, prop):
    E, s0, e_p, Y0, r_d, s_d = prop
    Y = energy_release_rate_Y(ee, xi_p, D, prop)
    q_d = r_d * xi_d
    return Y - (Y0 + q_d)


@jax.jit
def material_step_damage_plasticity(state, ep_new):
    ep_p_n, xi_p_n, D_n, xi_d_n = state

    E    = 3.0
    s0   = 0.6
    e_p  = 0.4
    Y0   = 0.15
    r_d  = 0.5
    s_d  = 0.05
    prop = (E, s0, e_p, Y0, r_d, s_d)

    tol = 1e-8
    max_iter = 50

    ep_p_tr = ep_p_n
    xi_p_tr = xi_p_n
    D_tr    = D_n
    xi_d_tr = xi_d_n

    ee_tr    = ep_new - ep_p_tr
    sigma_tr = f_dam(D_tr) ** 2 * E * ee_tr

    Fp_tr = yield_function_Fp(sigma_tr, xi_p_tr, D_tr, prop)
    Fd_tr = damage_function_Fd(ee_tr, xi_p_tr, xi_d_tr, D_tr, prop)

    def elastic_case(_):
        sigma = sigma_tr
        return (ep_p_n, xi_p_n, D_n, xi_d_n), sigma

    def plastic_only_case(_):
        sgn = jnp.where(sigma_tr >= 0.0, 1.0, -1.0)

        def plastic_iter(carry):
            Dlp, _, itr = carry
            fd       = jnp.maximum(f_dam(D_n), 1e-6)
            ep_p_new = ep_p_n + Dlp * sgn
            xi_p_new = jnp.maximum(xi_p_n + Dlp / fd, xi_p_n)
            ee_new    = ep_new - ep_p_new
            sigma_new = fd ** 2 * E * ee_new
            q_p = fd ** 2 * e_p * xi_p_new
            res = jnp.abs(sigma_new) - fd * s0 - q_p
            d_Fp_dDlp = -fd ** 2 * E - fd * e_p
            d_Fp_dDlp = jnp.where(jnp.abs(d_Fp_dDlp) < 1e-14, -1e-14, d_Fp_dDlp)
            Dlp_new = jnp.maximum(Dlp - res / d_Fp_dDlp, 0.0)
            return Dlp_new, jnp.abs(res), itr + 1

        def cond(carry):
            _, rn, itr = carry
            return (rn > tol) & (itr < max_iter)

        fd0 = f_dam(D_n)
        Dlp0 = jnp.maximum(Fp_tr / (fd0 ** 2 * E + fd0 * e_p), 0.0)
        Dlp_f, _, _ = jax.lax.while_loop(cond, plastic_iter, (Dlp0, 1.0, 0))

        fd = f_dam(D_n)
        ep_p_new = ep_p_n + Dlp_f * sgn
        xi_p_new = jnp.maximum(xi_p_n + Dlp_f / fd, xi_p_n)
        ee_new = ep_new - ep_p_new
        sigma = fd ** 2 * E * ee_new
        return (ep_p_new, xi_p_new, D_n, xi_d_n), sigma

    def damage_only_case(_):
        ee_new = ep_new - ep_p_n

        def damage_iter(carry):
            Dld, _, itr = carry
            D_new    = jnp.clip(D_n + Dld, 0.0, 0.999)
            xi_d_new = (xi_d_n + Dld) / (1.0 + s_d * Dld)
            xi_d_new = jnp.maximum(xi_d_new, xi_d_n)
            Y_new = energy_release_rate_Y(ee_new, xi_p_n, D_new, prop)
            q_d = r_d * xi_d_new
            res = Y_new - (Y0 + q_d)
            dY_dD = -(E * ee_new ** 2 + e_p * xi_p_n ** 2)
            dqd_dDld = r_d / (1.0 + s_d * Dld) ** 2
            d_Fd_dDld = dY_dD - dqd_dDld
            d_Fd_dDld = jnp.where(jnp.abs(d_Fd_dDld) < 1e-14, -1e-14, d_Fd_dDld)
            Dld_new = jnp.maximum(Dld - res / d_Fd_dDld, 0.0)
            return Dld_new, jnp.abs(res), itr + 1

        def cond(carry):
            _, rn, itr = carry
            return (rn > tol) & (itr < max_iter)

        dY_dD_est = -(E * ee_new ** 2 + e_p * xi_p_n ** 2)
        Dld0 = jnp.maximum(Fd_tr / (-dY_dD_est + r_d + 1e-8), 0.0)
        Dld_f, _, _ = jax.lax.while_loop(cond, damage_iter, (Dld0, 1.0, 0))

        D_new    = jnp.clip(D_n + Dld_f, 0.0, 0.999)
        xi_d_new = jnp.maximum((xi_d_n + Dld_f) / (1.0 + s_d * Dld_f), xi_d_n)
        sigma    = f_dam(D_new) ** 2 * E * ee_new
        return (ep_p_n, xi_p_n, D_new, xi_d_new), sigma

    def coupled_case(_):
        sgn = jnp.where(sigma_tr >= 0.0, 1.0, -1.0)

        def coupled_iter(carry):
            Dlp, Dld, _, itr = carry
            D_new    = jnp.clip(D_n + Dld, 0.0, 0.999)
            fd       = jnp.maximum(f_dam(D_new), 1e-3)
            ep_p_new = ep_p_n + Dlp * sgn
            dxi_p    = jnp.minimum(Dlp / fd, 10.0)
            xi_p_new = jnp.maximum(xi_p_n + dxi_p, xi_p_n)
            xi_d_new = jnp.maximum((xi_d_n + Dld) / (1.0 + s_d * Dld), xi_d_n)
            ee_new   = ep_new - ep_p_new
            sigma    = fd ** 2 * E * ee_new
            q_p = fd ** 2 * e_p * xi_p_new
            Fp  = jnp.abs(sigma) - fd * s0 - q_p
            Y   = energy_release_rate_Y(ee_new, xi_p_new, D_new, prop)
            q_d = r_d * xi_d_new
            Fd  = Y - (Y0 + q_d)
            res_norm = jnp.sqrt(Fp ** 2 + Fd ** 2)

            dFp_dDlp = -fd ** 2 * E - fd * e_p
            dFp_dDld = -2.0 * fd * E * jnp.abs(ee_new) + s0 + 2.0 * fd * e_p * xi_p_new
            dY_dee   = (1.0 - D_new) * 2.0 * E * ee_new
            dY_dxip  = (1.0 - D_new) * 2.0 * e_p * xi_p_new
            dFd_dDlp = dY_dee * (-sgn) + dY_dxip * (1.0 / fd)
            dY_dD    = -(E * ee_new ** 2 + e_p * xi_p_new ** 2)
            dqd_dDld = r_d / (1.0 + s_d * Dld) ** 2
            dFd_dDld = dY_dD - dqd_dDld

            J   = jnp.array([[dFp_dDlp, dFp_dDld],
                              [dFd_dDlp, dFd_dDld]])
            rhs = jnp.array([-Fp, -Fd])
            J = J + jnp.eye(2) * 1e-14
            delta = jnp.linalg.solve(J, rhs)

            Dlp_new = jnp.maximum(Dlp + delta[0], 0.0)
            Dld_new = jnp.maximum(Dld + delta[1], 0.0)
            return Dlp_new, Dld_new, res_norm, itr + 1

        def cond(carry):
            _, _, rn, itr = carry
            return (rn > tol) & (itr < max_iter)

        fd0  = f_dam(D_n)
        Dlp0 = jnp.maximum(Fp_tr / (fd0 ** 2 * E + fd0 * e_p), 0.0)
        dY0  = -(E * ee_tr ** 2 + e_p * xi_p_n ** 2)
        Dld0 = jnp.maximum(Fd_tr / (-dY0 + r_d + 1e-8), 0.0)

        Dlp_f, Dld_f, _, _ = jax.lax.while_loop(
            cond, coupled_iter, (Dlp0, Dld0, 1.0, 0))

        D_new    = jnp.clip(D_n + Dld_f, 0.0, 0.999)
        fd       = jnp.maximum(f_dam(D_new), 1e-3)
        ep_p_new = ep_p_n + Dlp_f * sgn
        dxi_p    = jnp.minimum(Dlp_f / fd, 10.0)
        xi_p_new = jnp.maximum(xi_p_n + dxi_p, xi_p_n)
        xi_d_new = jnp.maximum((xi_d_n + Dld_f) / (1.0 + s_d * Dld_f), xi_d_n)
        ee_new   = ep_new - ep_p_new
        sigma    = fd ** 2 * E * ee_new
        return (ep_p_new, xi_p_new, D_new, xi_d_new), sigma

    plastic_active = Fp_tr > 0.0
    damage_active  = Fd_tr > 0.0

    def no_plastic_branch(_):
        return jax.lax.cond(damage_active, damage_only_case, elastic_case, None)

    def plastic_branch(_):
        return jax.lax.cond(damage_active, coupled_case, plastic_only_case, None)

    new_state, sigma = jax.lax.cond(plastic_active, plastic_branch,
                                    no_plastic_branch, None)
    return new_state, sigma


@jax.jit
def solve_full_path(strain_history):
    initial_state = (0.0, 0.0, 0.0, 0.0)

    def scan_body(carry, strain):
        new_state, stress = material_step_damage_plasticity(carry, strain)
        return new_state, stress

    _, stress_history = jax.lax.scan(scan_body, initial_state, strain_history)
    return stress_history


# ==================
# 2. DATA GENERATORS 
# ==================
def generate_gp_paths(num_samples, num_steps, max_strain=1.0):
    print(f"Generating {num_samples} GP paths...")
    t = np.linspace(0, 1, num_steps).reshape(-1, 1)
    strain_data = np.zeros((num_samples, num_steps))
    num_batches = 10
    batch_size = num_samples // num_batches
    for i in range(num_batches):
        length_scale = np.random.uniform(0.005, 0.05)
        dists = cdist(t, t, metric='sqeuclidean')
        K = np.exp(-dists / (2 * length_scale ** 2)) + np.eye(num_steps) * 1e-6
        batch_paths = np.random.multivariate_normal(np.zeros(num_steps), K, batch_size)
        batch_paths -= batch_paths[:, 0:1]
        max_vals = np.max(np.abs(batch_paths), axis=1, keepdims=True)
        max_vals[max_vals < 1e-6] = 1.0
        target_scales = np.random.uniform(0.1, max_strain, size=(batch_size, 1))
        batch_paths = (batch_paths / max_vals) * target_scales
        s, e = i * batch_size, (i + 1) * batch_size
        if i == num_batches - 1:
            strain_data[s:] = batch_paths[:num_samples - s]
        else:
            strain_data[s:e] = batch_paths
    return jnp.array(strain_data[..., None])


def generate_random_zigzag_paths(num_samples, num_steps, max_strain=1.0):
    key_points = np.linspace(0, num_steps - 1, 7, dtype=int)
    strain_data = np.zeros((num_samples, num_steps))
    for i in range(num_samples):
        checkpoints = np.random.uniform(-max_strain, max_strain, size=5)
        values = np.concatenate(([0], checkpoints, [0]))
        strain_data[i, :] = np.interp(np.arange(num_steps), key_points, values)
    print(f"Generated {num_samples} zig-zag paths.")
    return jnp.array(strain_data[..., None])


def generate_sinusoidal_paths(num_samples, num_steps, max_strain=1.0, T_total=1.0):
    print(f"Generating {num_samples} sinusoidal paths...")
    t = np.linspace(0, T_total, num_steps)
    strain_data = np.zeros((num_samples, num_steps))
    for i in range(num_samples):
        freq = np.random.uniform(1, 5)
        amplitude = np.random.uniform(0.1, max_strain)
        strain_data[i, :] = amplitude * np.abs(np.sin(2 * np.pi * freq * t / T_total))
        strain_data[i, 0] = 0.0
    return jnp.array(strain_data[..., None])


# ============================================================
# 3. MODEL: FNO + SIREN + CAUSAL ATTENTION 
# ============================================================



class SirenLinear(nnx.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0, rngs: nnx.Rngs = None):
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega0 = omega0
        key = rngs.params()
        if is_first:
            limit = 1.0 / in_features
            w_init = jax.random.uniform(key, (in_features, out_features), minval=-limit, maxval=limit)
        else:
            limit = jnp.sqrt(6.0 / in_features) / omega0
            w_init = jax.random.uniform(key, (in_features, out_features), minval=-limit, maxval=limit)
        self.w = nnx.Param(w_init)
        self.b = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x):
        x = x @ self.w.value + self.b.value
        if self.is_first:
            x = self.omega0 * x
        return x


class SpectralConv1D(nnx.Module):
    def __init__(self, in_channels, out_channels, modes, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nnx.Param(jax.random.uniform(rngs.params(),
                                                    (in_channels, out_channels, modes, 2),
                                                    minval=-scale, maxval=scale))

    def __call__(self, x):
        B, T, C = x.shape
        x_ft = jnp.fft.rfft(x, axis=1)
        actual_modes = min(self.modes, T // 2 + 1)
        total_freqs = T // 2 + 1
        w_complex = self.weights.value[..., 0] + 1j * self.weights.value[..., 1]
        x_ft_trunc = x_ft[:, :actual_modes, :]
        w_trunc = w_complex[:, :, :actual_modes]
        out_ft = jnp.einsum("bmi,iom->bmo", x_ft_trunc, w_trunc)
        out_ft_padded = jnp.zeros((B, total_freqs, self.out_channels), dtype=jnp.complex64)
        out_ft_padded = out_ft_padded.at[:, :actual_modes, :].set(out_ft)
        x = jnp.fft.irfft(out_ft_padded, n=T, axis=1)
        return x


class CausalMultiHeadSelfAttention(nnx.Module):
    """Multi-head self-attention with strict causal mask. Output at position t
    depends only on inputs at positions <= t."""
    def __init__(self, dim, num_heads=4, dropout_rate=0.1, rngs: nnx.Rngs = None):
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.attn_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.proj_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, training=False):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))                 # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn_logits = jnp.where(causal_mask[None, None, :, :], attn_logits,
                                jnp.finfo(attn_logits.dtype).min)

        attn = jax.nn.softmax(attn_logits, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not training)

        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)

        out = self.proj(out)
        out = self.proj_dropout(out, deterministic=not training)
        return out


class FNOAttnBlock(nnx.Module):
    """One block: pre-norm causal attention (residual) + FNO sub-block
    (spectral conv + pointwise SIREN, sin activation, dropout)."""
    def __init__(self, width, modes, omega0=30.0, num_heads=4,
                 dropout_rate=0.1, rngs: nnx.Rngs = None):
        self.spectral = SpectralConv1D(width, width, modes, rngs)
        self.w = SirenLinear(width, width, is_first=False, omega0=omega0, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(width, rngs=rngs)
        self.attn = CausalMultiHeadSelfAttention(
            dim=width, num_heads=num_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, training=False):
        # 1. Causal attention with residual + pre-norm.
        x = x + self.attn(self.attn_norm(x), training=training)
        # 2. FNO sub-block: spectral + pointwise SIREN, sin activation.
        x_spec = self.spectral(x)
        x_loc = self.w(x)
        x = jnp.sin(x_spec + x_loc)
        x = self.dropout(x, deterministic=not training)
        return x


class FNO1D_SIREN_Attn(nnx.Module):
    """Attention-augmented FNO+SIREN. Causal mask enforces stress(t) depends
    only on strain(<=t), respecting plasticity's path dependence."""
    def __init__(self, modes, width, depth, rngs: nnx.Rngs, padding_frac=0.1,
                 omega0=30.0, num_heads=4, dropout_rate=0.1):
        self.width = width
        self.modes = modes
        self.padding_frac = padding_frac
        lifting_in_dim = 1

        self.fc0 = SirenLinear(lifting_in_dim, width, is_first=True,
                               omega0=omega0, rngs=rngs)

        # FNO + Attention blocks
        self.blocks = []
        for _ in range(depth):
            self.blocks.append(
                FNOAttnBlock(width=width, modes=modes, omega0=omega0,
                             num_heads=num_heads, dropout_rate=dropout_rate,
                             rngs=rngs)
            )

        # Projection
        self.fc1 = SirenLinear(width, 128, is_first=False, omega0=omega0, rngs=rngs)
        self.fc2 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x, training=False):
        B, T, C = x.shape


        # Lifting.
        x = self.fc0(x)
        x = jnp.sin(x)

        # Pad before blocks (causal mask still strictly lower-triangular).
        n_pad = int(T * self.padding_frac)
        if n_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)))

        # Blocks.
        for block in self.blocks:
            x = block(x, training=training)

        # Crop padding.
        if n_pad > 0:
            x = x[:, :T, :]

        # Projection.
        x = self.fc1(x)
        x = jnp.sin(x)
        x = self.fc2(x)

        # Enforce stress(t=0) = 0.
        x = x - x[:, 0:1, :]
        return x


# ============================================================
# 4. SAVE MODEL 
# ============================================================
def save_model(model, model_config, filepath):
    flat = {}

    def extract(obj, prefix=""):
        for attr in list(vars(obj).keys()):
            val = getattr(obj, attr)
            path = f"{prefix}.{attr}" if prefix else attr
            if isinstance(val, nnx.Variable):
                raw = getattr(val, 'raw_value', None)
                if raw is not None:
                    flat[path] = np.array(raw)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, nnx.Module):
                        extract(item, f"{path}[{i}]")
                    elif isinstance(item, nnx.Variable):
                        raw = getattr(item, 'raw_value', None)
                        if raw is not None:
                            flat[f"{path}[{i}]"] = np.array(raw)
            elif isinstance(val, nnx.Module):
                extract(val, path)

    extract(model)
    with open(filepath, 'wb') as f:
        pickle.dump({'flat_params': flat, 'model_config': model_config}, f)
    print(f"   ✓ Model saved to {filepath} ({len(flat)} arrays)")


# ============================================================
# 5. BUILD + TRAIN HELPERS
# ============================================================
def build_model(modes, width, depth, omega0, num_heads, dropout_rate, seed=0):
    return FNO1D_SIREN_Attn(
        modes=modes, width=width, depth=depth,
        rngs=nnx.Rngs(seed), omega0=omega0,
        num_heads=num_heads, dropout_rate=dropout_rate,
    )


def train_model(model, opt_state, optimizer,
                x_train, y_train, x_val, y_val, x_test, y_test,
                batch_size, epochs, patience,
                trial=None, log_every=50, record_history=True, verbose=True):

    @nnx.jit
    def train_step(model, opt_state, bx, by):
        def loss_fn(m):
            return jnp.mean((m(bx, training=True) - by) ** 2)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        params = nnx.state(model, nnx.Param)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return loss, new_opt_state

    @nnx.jit
    def eval_step(model, bx, by):
        return jnp.mean((model(bx, training=False) - by) ** 2)

    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    n_test = x_test.shape[0]
    n_train_b = n_train // batch_size
    n_val_b = math.ceil(n_val / batch_size)
    n_test_b = math.ceil(n_test / batch_size)

    best_val = float('inf')
    best_params = None
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': [], 'test': []}

    x_train_j = jnp.array(x_train)
    y_train_j = jnp.array(y_train)

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        x_s = x_train_j[perm]
        y_s = y_train_j[perm]

        train_loss = 0.0
        for i in range(n_train_b):
            bx = x_s[i * batch_size:(i + 1) * batch_size]
            by = y_s[i * batch_size:(i + 1) * batch_size]
            loss, opt_state = train_step(model, opt_state, bx, by)
            train_loss += loss
        train_loss /= max(n_train_b, 1)

        val_loss = 0.0
        for i in range(n_val_b):
            bx = x_val[i * batch_size:(i + 1) * batch_size]
            by = y_val[i * batch_size:(i + 1) * batch_size]
            val_loss += eval_step(model, bx, by)
        val_loss /= max(n_val_b, 1)

        if record_history:
            test_loss = 0.0
            for i in range(n_test_b):
                bx = x_test[i * batch_size:(i + 1) * batch_size]
                by = y_test[i * batch_size:(i + 1) * batch_size]
                test_loss += eval_step(model, bx, by)
            test_loss /= max(n_test_b, 1)
            history['train'].append(float(train_loss))
            history['val'].append(float(val_loss))
            history['test'].append(float(test_loss))

        val_f = float(val_loss)
        if val_f < best_val:
            best_val = val_f
            best_params = nnx.state(model)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % log_every == 0:
            extra = f" | test {history['test'][-1]:.5f}" if record_history else ""
            print(f"  Epoch {epoch:5d} | train {float(train_loss):.5f} | val {val_f:.5f}{extra}")

        if trial is not None:
            trial.report(val_f, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_counter >= patience:
            if verbose:
                print(f"  !! Early stop at epoch {epoch}. Best val {best_val:.6f}")
            break

    return best_val, best_params, history, best_epoch, opt_state


# ============================================================
# 6. TEST SUITE 
# ============================================================
def sanity_check_material_model(results_dir, num_steps=200):
    print("\n--- SANITY CHECK: Brepols damage-plasticity ---")
    t = np.linspace(0, 1, num_steps)
    path_mono = np.linspace(0, 1.5, num_steps)
    path_cyclic = 0.8 * np.sin(2 * np.pi * 2 * t); path_cyclic[0] = 0.0
    path_elastic = 0.1 * np.sin(2 * np.pi * t); path_elastic[0] = 0.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, (label, path) in enumerate([
        ("Monotonic Tension", path_mono),
        ("Cyclic (large)",    path_cyclic),
        ("Elastic only",      path_elastic),
    ]):
        stress = np.array(solve_full_path(jnp.array(path)))
        state = (0.0, 0.0, 0.0, 0.0)
        damage = []
        for ep_val in path:
            state, _ = material_step_damage_plasticity(state, float(ep_val))
            damage.append(float(state[2]))
        damage = np.array(damage)

        axes[0, col].plot(t, stress, 'b-', linewidth=2)
        axes[0, col].set_xlabel('Time'); axes[0, col].set_ylabel('Stress [MPa]')
        axes[0, col].set_title(f'{label}\nStress vs Time'); axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(t, damage, 'r-', linewidth=2)
        axes[1, col].set_ylim(-0.05, 1.05)
        axes[1, col].set_xlabel('Time'); axes[1, col].set_ylabel('Damage D')
        axes[1, col].set_title(f'Damage\n(max D={damage[-1]:.4f})')
        axes[1, col].grid(True, alpha=0.3)
        print(f"  {label}: max |sigma|={np.max(np.abs(stress)):.4f}, final D={damage[-1]:.4f}")

    plt.suptitle('Brepols (2017) Sanity Check', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/sanity_check_damage_plasticity.png", dpi=150)
    plt.close()


def comprehensive_statistical_testing(model, num_steps, results_dir):
    print("\n=== COMPREHENSIVE STATISTICAL TESTING ===")
    T_TOTAL = 1.0
    time_axis = np.linspace(0, T_TOTAL, num_steps)
    batch_solver = jax.vmap(solve_full_path)

    def run(name, inputs):
        chunk = 20
        truth_list = []
        for i in range(0, 100, chunk):
            truth_list.append(batch_solver(inputs[i:i + chunk, :, 0]))
        truth = jnp.concatenate(truth_list, axis=0)[..., None]
        pred = model(inputs, training=False)
        errs = []
        for i in range(100):
            e = jnp.sqrt(jnp.mean((pred[i] - truth[i]) ** 2)) / (
                jnp.sqrt(jnp.mean(truth[i] ** 2)) + 1e-12)
            errs.append(float(e))
        print(f"  {name}: mean rel L2 = {np.mean(errs):.6f} ± {np.std(errs):.6f}")
        return errs, inputs, truth, pred

    zz_errs, zz_in, zz_tr, zz_pr = run("Zig-Zag",
                                       generate_random_zigzag_paths(100, num_steps, 1.0))
    sn_errs, sn_in, sn_tr, sn_pr = run("Sinusoidal",
                                       generate_sinusoidal_paths(100, num_steps, 1.0, T_TOTAL))
    gp_errs, gp_in, gp_tr, gp_pr = run("Gaussian Process",
                                       generate_gp_paths(100, num_steps, 1.0))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, max(max(zz_errs), max(sn_errs), max(gp_errs)), 30)
    plt.hist(zz_errs, bins=bins, alpha=0.6, label='Zig-Zag', color='blue', edgecolor='black')
    plt.hist(sn_errs, bins=bins, alpha=0.6, label='Sinusoidal', color='red', edgecolor='black')
    plt.hist(gp_errs, bins=bins, alpha=0.6, label='GP', color='green', edgecolor='black')
    plt.xlabel('Relative L2 Error'); plt.ylabel('Frequency')
    plt.title('Error Distributions (n=100)'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    bp = plt.boxplot([zz_errs, sn_errs, gp_errs],
                     labels=['Zig-Zag', 'Sinusoidal', 'GP'],
                     patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(c)
    plt.ylabel('Relative L2 Error'); plt.title('Box Plot'); plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comprehensive_error_distribution.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name, errs, color in zip(axes,
                                     ['Zig-Zag', 'Sinusoidal', 'GP'],
                                     [zz_errs, sn_errs, gp_errs],
                                     ['blue', 'red', 'green']):
        ax.plot(range(1, 101), sorted(errs), 'o-', color=color, alpha=0.6, markersize=3)
        ax.axhline(np.mean(errs), color='black', linestyle='--', linewidth=2,
                   label=f'mean = {np.mean(errs):.5f}')
        ax.fill_between(range(1, 101),
                        np.mean(errs) - np.std(errs),
                        np.mean(errs) + np.std(errs),
                        alpha=0.2, color=color, label=f'±1σ = {np.std(errs):.5f}')
        ax.set_xlabel('Sample (sorted)'); ax.set_ylabel('Rel L2')
        ax.set_title(f'{name} (n=100)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/individual_error_distributions.png", dpi=150)
    plt.close()

    with open(f"{results_dir}/statistical_summary.txt", 'w') as f:
        f.write("=" * 70 + "\nCOMPREHENSIVE STATISTICAL TESTING RESULTS\n")
        f.write("Brepols et al. (2017) Damage-Plasticity + FNO-SIREN-Attention\n")
        f.write("=" * 70 + "\n\n")
        for name, errs in [("ZIG-ZAG", zz_errs), ("SINUSOIDAL", sn_errs),
                           ("GAUSSIAN PROCESS", gp_errs)]:
            f.write(f"{name} (n=100)\n" + "-" * 50 + "\n")
            f.write(f"  Mean:   {np.mean(errs):.8f}\n")
            f.write(f"  Std:    {np.std(errs):.8f}\n")
            f.write(f"  Min:    {np.min(errs):.8f}\n")
            f.write(f"  Max:    {np.max(errs):.8f}\n")
            f.write(f"  Median: {np.median(errs):.8f}\n")
            f.write(f"  25%:    {np.percentile(errs, 25):.8f}\n")
            f.write(f"  75%:    {np.percentile(errs, 75):.8f}\n")
            f.write(f"  95%:    {np.percentile(errs, 95):.8f}\n\n")

    for cat, inp, tr, pr in [('zigzag', zz_in, zz_tr, zz_pr),
                             ('sinusoidal', sn_in, sn_tr, sn_pr),
                             ('gp', gp_in, gp_tr, gp_pr)]:
        idxs = np.random.choice(100, 3, replace=False)
        for i, idx in enumerate(idxs):
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(time_axis, inp[idx].flatten(), 'b-', linewidth=2)
            plt.title(f'{cat.upper()} {idx}: Strain'); plt.grid(True, alpha=0.3)
            plt.subplot(2, 2, 2)
            plt.plot(time_axis, tr[idx].flatten(), 'g-', linewidth=2, label='Truth')
            plt.plot(time_axis, pr[idx].flatten(), 'r--', linewidth=2, label='FNO')
            plt.legend(); plt.title('Stress'); plt.grid(True, alpha=0.3)
            plt.subplot(2, 2, 3)
            plt.plot(inp[idx].flatten(), tr[idx].flatten(), 'g-', linewidth=2, label='Truth')
            plt.plot(inp[idx].flatten(), pr[idx].flatten(), 'r--', linewidth=2, label='Pred')
            plt.legend(); plt.title('Hysteresis'); plt.grid(True, alpha=0.3)
            plt.subplot(2, 2, 4)
            err = np.abs(tr[idx].flatten() - pr[idx].flatten())
            rl2 = np.sqrt(np.mean(err ** 2)) / (np.sqrt(np.mean(tr[idx].flatten() ** 2)) + 1e-12)
            plt.plot(time_axis, err, 'orange', linewidth=2)
            plt.axhline(np.mean(err), color='red', linestyle='--',
                        label=f'Rel L2 = {rl2:.5f}')
            plt.legend(); plt.title('Error'); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/comprehensive_{cat}_sample_{i}.png", dpi=150)
            plt.close()

    return {'zigzag': zz_errs, 'sinusoidal': sn_errs, 'gp': gp_errs}


def generate_canonical_zigzag_paths(num_samples, num_key_points=7,
                                    max_strain=1.0, seed=42):
    np.random.seed(seed)
    key_points = np.linspace(0, 1, num_key_points)
    key_values = []
    for _ in range(num_samples):
        interior = np.random.uniform(-max_strain, max_strain, size=num_key_points - 2)
        end = np.random.uniform(0, max_strain)
        key_values.append(np.concatenate(([0], interior, [end])))
    return key_points, np.array(key_values)


def generate_resampled_path(key_points, key_values, num_steps):
    step_pos = np.linspace(0, 1, num_steps)
    return np.interp(step_pos, key_points, key_values)


def test_discretization_sensitivity(model, results_dir, resolutions,
                                    num_test_samples, training_resolution, seed=12345):
    print("\n=== DISCRETIZATION SENSITIVITY ===")
    print(f"Resolutions: {resolutions}, training_res = {training_resolution}")

    key_points, key_values = generate_canonical_zigzag_paths(
        num_test_samples, 7, 1.0, seed=seed)

    results = {'resolutions': resolutions, 'training_resolution': training_resolution,
               'mean_error': [], 'std_error': [], 'individual_errors': {}}

    for r_idx, n_steps in enumerate(resolutions):
        print(f"  [{r_idx+1}/{len(resolutions)}] resolution {n_steps} ...")
        strains = np.zeros((num_test_samples, n_steps))
        for i in range(num_test_samples):
            strains[i] = generate_resampled_path(key_points, key_values[i], n_steps)
        strains_j = jnp.array(strains[..., None])
        bs = jax.vmap(solve_full_path)
        truth = bs(strains_j[..., 0])
        pred = model(strains_j, training=False)[..., 0]

        ind = []
        for i in range(num_test_samples):
            mse_i = np.mean((pred[i] - truth[i]) ** 2)
            rms = np.sqrt(np.mean(truth[i] ** 2))
            ind.append(float(np.sqrt(mse_i) / rms) if rms > 1e-10 else float(np.sqrt(mse_i)))

        results['mean_error'].append(np.mean(ind))
        results['std_error'].append(np.std(ind))
        results['individual_errors'][n_steps] = ind
        print(f" Mean: {np.mean(ind)*100:.2f}%")

    results['mean_error'] = np.array(results['mean_error'])
    results['std_error'] = np.array(results['std_error'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1 = axes[0, 0]
    ax1.plot(resolutions, results['mean_error'] * 100, 'bo-', linewidth=2, markersize=8, label='NRMSE')
    ax1.fill_between(resolutions,
                     (results['mean_error'] - results['std_error']) * 100,
                     (results['mean_error'] + results['std_error']) * 100,
                     alpha=0.3, color='blue', label='±1σ')
    if training_resolution in resolutions:
        ti = resolutions.index(training_resolution)
        ax1.axvline(training_resolution, color='red', linestyle='--', linewidth=2,
                    label=f'Training ({training_resolution})')
        ax1.scatter([training_resolution], [results['mean_error'][ti] * 100],
                    color='red', s=150, zorder=5, marker='*')
    ax1.set_xlabel('Temporal Resolution'); ax1.set_ylabel('MSE (%)')
    ax1.set_title('Discretization Sensitivity'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    box_data = [results['individual_errors'][r] for r in resolutions]
    bp = ax2.boxplot(box_data, labels=[str(r) for r in resolutions],
                     patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], plt.cm.coolwarm(np.linspace(0, 1, len(resolutions)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_xlabel('Resolution'); ax2.set_ylabel('Rel L2')
    ax2.set_title('Error by Resolution'); ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[1, 0]
    if training_resolution in resolutions:
        ti = resolutions.index(training_resolution)
        base = results['mean_error'][ti]
        rel_change = (results['mean_error'] - base) / base * 100
        colors_bar = ['green' if rc <= 0 else 'red' for rc in rel_change]
        ax3.bar(range(len(resolutions)), rel_change, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(resolutions)))
        ax3.set_xticklabels([str(r) for r in resolutions], rotation=45)
        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_xlabel('Resolution'); ax3.set_ylabel('Rel change in Mean Error (%)')
        ax3.set_title(f'Change vs training res ({training_resolution})')
        ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    em = np.array([results['individual_errors'][r] for r in resolutions]).T
    im = ax4.imshow(em, aspect='auto', cmap='YlOrRd',
                    extent=[0, len(resolutions), num_test_samples, 0])
    ax4.set_xticks(np.arange(len(resolutions)) + 0.5)
    ax4.set_xticklabels([str(r) for r in resolutions], rotation=45)
    ax4.set_xlabel('Resolution'); ax4.set_ylabel('Sample index')
    ax4.set_title('Individual Errors Heatmap')
    plt.colorbar(im, ax=ax4, label='Rel L2')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/discretization_sensitivity.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    show = []
    if len(resolutions) >= 6:
        ids = [0, len(resolutions)//5, 2*len(resolutions)//5,
               3*len(resolutions)//5, 4*len(resolutions)//5, -1]
        show = [resolutions[i] for i in ids]
    else:
        show = resolutions[:6]
    if training_resolution in resolutions and training_resolution not in show and len(show) >= 3:
        show[2] = training_resolution

    for a_i, n_steps in enumerate(show[:6]):
        ax = axes[a_i]
        strain = generate_resampled_path(key_points, key_values[0], n_steps)
        stress = np.array(solve_full_path(jnp.array(strain)))
        pred = np.array(model(jnp.array(strain.reshape(1, n_steps, 1)), training=False)[0, :, 0])
        ax.plot(strain, stress, 'b-', linewidth=2, label='Truth')
        ax.plot(strain, pred, 'r--', linewidth=2, label='FNO')
        mse_i = np.mean((pred - stress) ** 2)
        suffix = ' (TRAINING)' if n_steps == training_resolution else ''
        color = 'green' if n_steps == training_resolution else 'black'
        ax.set_title(f'{n_steps} steps{suffix}\nMSE {mse_i*100:.2f}%',
                     color=color, fontweight='bold')
        ax.set_xlabel('Strain'); ax.set_ylabel('Stress [MPa]')
        ax.legend(); ax.grid(True, alpha=0.3)
    for a_i in range(len(show), len(axes)):
        axes[a_i].axis('off')
    plt.suptitle('Same physical path, different resolutions', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/discretization_example_predictions.png", dpi=150)
    plt.close()

    with open(f"{results_dir}/discretization_sensitivity_summary.txt", 'w') as f:
        f.write("=" * 70 + "\nDISCRETIZATION SENSITIVITY (Brepols + Attention)\n"
                + "=" * 70 + "\n")
        f.write(f"Training resolution: {training_resolution}\n")
        f.write(f"Test samples: {num_test_samples}\n")
        f.write(f"Resolutions: {resolutions}\n\n")
        f.write(f"{'Res':<10}{'Mean (%)':<12}{'Std (%)':<10}\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(resolutions):
            mk = ' <-- TRAIN' if r == training_resolution else ''
            f.write(f"{r:<10}{results['mean_error'][i]*100:<12.2f}"
                    f"{results['std_error'][i]*100:<10.2f}{mk}\n")
    return results


def test_xsinx_case(model, N_STEPS, T_TOTAL, results_dir):
    print("\n=== x*sin(x) TEST ===")
    x = np.linspace(0, 3 * np.pi, N_STEPS)
    raw = x * np.sin(x)
    raw = raw - raw[0]
    if np.max(np.abs(raw)) > 1e-8:
        raw = raw / np.max(np.abs(raw))
    strain = raw
    t = np.linspace(0, T_TOTAL, N_STEPS)
    gt = np.array(solve_full_path(jnp.array(strain)))
    pr = np.array(model(jnp.array(strain.reshape(1, N_STEPS, 1)), training=False)[0, :, 0])
    rl2 = np.sqrt(np.mean((pr - gt) ** 2)) / (np.sqrt(np.mean(gt ** 2)) + 1e-12)
    print(f"  x*sin(x) rel L2 = {rl2:.6f}")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, strain, 'b-', linewidth=2); ax[0, 0].set_title("x·sin(x) Strain")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 1].plot(t, gt, 'g-', linewidth=2, label='Truth')
    ax[0, 1].plot(t, pr, 'r--', linewidth=2, label='FNO')
    ax[0, 1].set_title("Stress"); ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)
    ax[1, 0].plot(strain, gt, 'g-', linewidth=2, label='Truth')
    ax[1, 0].plot(strain, pr, 'r--', linewidth=2, label='Pred')
    ax[1, 0].set_title("Hysteresis"); ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)
    err = np.abs(gt - pr)
    ax[1, 1].plot(t, err, 'orange', linewidth=2)
    ax[1, 1].axhline(np.mean(err), color='red', linestyle='--',
                     label=f'Rel L2 = {rl2:.5f}')
    ax[1, 1].set_title("Abs Error"); ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/prediction_xsinx.png", dpi=150)
    plt.close()
    return rl2


def save_timing_info(results_dir, training_time, test_time, num_test_samples,
                     history, best_epoch):
    with open(f"{results_dir}/timing_info.txt", 'w') as f:
        f.write("=" * 70 + "\nTIMING\n" + "=" * 70 + "\n\n")
        f.write("TRAINING\n" + "-" * 70 + "\n")
        f.write(f"  total: {training_time:.2f} s ({training_time/60:.2f} min)\n")
        f.write(f"  epochs: {len(history['train'])}, best {best_epoch + 1}\n")
        if len(history['train']) > 0:
            f.write(f"  avg/epoch: {training_time/len(history['train']):.2f} s\n")
            f.write(f"  best val: {min(history['val']):.6f}\n\n")
        f.write("TESTING\n" + "-" * 70 + "\n")
        f.write(f"  samples: {num_test_samples}\n")
        f.write(f"  total: {test_time:.2f} s\n")
        f.write(f"  per sample: {test_time/num_test_samples*1000:.2f} ms\n\n")
        f.write(f"TOTAL: {training_time + test_time:.2f} s\n")


# ============================================================
# 7. OPTUNA + MAIN
# ============================================================
RESULTS_DIR = "Material_operator_1D_coupled_plastic_damage"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_STEPS  = 50
N_GP     = 10000
N_TEST   = 2000
T_TOTAL  = 1.0


TUNE_EPOCHS   = 2000
TUNE_PATIENCE = 300
FINAL_EPOCHS   = 10000
FINAL_PATIENCE = 2000


N_TRIALS    = 100
OPTUNA_SEED = 42

# Joint (width, num_heads) options — every width is divisible by every paired
# head count, so Optuna can never propose an invalid combination.
WIDTH_HEADS_OPTIONS = [
    (16, 2), (16, 4),
    (32, 2), (32, 4), (32, 8),
    (48, 2), (48, 4), (48, 6),
    (64, 2), (64, 4), (64, 8),
    (96, 2), (96, 4), (96, 8),
]

_DATA = {}


def prepare_data():
    print("--- Generating data (once, reused for all trials) ---")
    np.random.seed(0)
    full_train_x = generate_gp_paths(N_GP, N_STEPS, 1.0)
    test_x = generate_random_zigzag_paths(N_TEST, N_STEPS, 1.0)

    bs = jax.vmap(solve_full_path)
    chunk = 1000

    y_list = []
    for i in range(0, full_train_x.shape[0], chunk):
        y_list.append(bs(full_train_x[i:i + chunk, :, 0]))
    full_train_y = jnp.concatenate(y_list, axis=0)[..., None]

    ty_list = []
    for i in range(0, test_x.shape[0], chunk):
        ty_list.append(bs(test_x[i:i + chunk, :, 0]))
    test_y = jnp.concatenate(ty_list, axis=0)[..., None]

    idx = np.arange(full_train_x.shape[0])
    np.random.shuffle(idx)
    split = int(0.9 * full_train_x.shape[0])
    _DATA['x_train'] = full_train_x[idx[:split]]
    _DATA['y_train'] = full_train_y[idx[:split]]
    _DATA['x_val']   = full_train_x[idx[split:]]
    _DATA['y_val']   = full_train_y[idx[split:]]
    _DATA['x_test']  = test_x
    _DATA['y_test']  = test_y
    print(f"  train {_DATA['x_train'].shape} | val {_DATA['x_val'].shape} | "
          f"test {_DATA['x_test'].shape}")


def objective(trial):
    # 8 hyperparameters: 7 from Code 1 + num_heads (paired with width).
    modes         = trial.suggest_categorical("modes",       [4, 8, 12, 16, 20, 24])
    wh_idx        = trial.suggest_categorical(
        "width_heads_idx", list(range(len(WIDTH_HEADS_OPTIONS))))
    width, num_heads = WIDTH_HEADS_OPTIONS[wh_idx]
    depth         = trial.suggest_int(        "depth",       4, 6)
    lr            = trial.suggest_float(      "lr",          1e-4, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size",  [32, 64, 128, 256])
    weight_decay  = trial.suggest_float(      "weight_decay",1e-6, 1e-2, log=True)
    omega0        = trial.suggest_float(      "omega0",      5.0, 30.0)
    dropout_rate  = trial.suggest_float(      "dropout_rate",0.0, 0.2)

    # Make tuned values visible in Optuna's UI as user attrs.
    trial.set_user_attr("width", width)
    trial.set_user_attr("num_heads", num_heads)

    modes = min(modes, N_STEPS // 2 + 1)

    print(f"\n[Trial {trial.number}] modes={modes} width={width} heads={num_heads} "
          f"depth={depth} lr={lr:.2e} bs={batch_size} wd={weight_decay:.2e} "
          f"omega0={omega0:.1f} dr={dropout_rate:.2f}")

    model = build_model(modes, width, depth, omega0, num_heads, dropout_rate,
                        seed=trial.number)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))

    try:
        best_val, _, _, _, _ = train_model(
            model, opt_state, optimizer,
            _DATA['x_train'], _DATA['y_train'],
            _DATA['x_val'],   _DATA['y_val'],
            _DATA['x_test'],  _DATA['y_test'],
            batch_size=batch_size,
            epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
            trial=trial, log_every=200,
            record_history=False, verbose=True,
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial crashed: {e}")
        return float('inf')

    return best_val


def main():
    t0 = time_module.time()
    print("Brepols et al. (2017) Damage-Plasticity + FNO-SIREN-Attention — Optuna tuning")

    sanity_check_material_model(RESULTS_DIR, num_steps=200)
    prepare_data()

    # -------------- OPTUNA STUDY --------------
    print(f"\n--- Running Optuna ({N_TRIALS} trials, MEDIUM budget) ---")
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=200),
        study_name="fno_brepols_attention",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   gc_after_trial=True)

    print("\n--- Optuna done ---")
    print(f"Best value (val MSE): {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Best user attrs: {study.best_trial.user_attrs}")

    with open(f"{RESULTS_DIR}/optuna_best_params.json", 'w') as f:
        json.dump({'best_value': study.best_value,
                   'best_params': study.best_params,
                   'best_user_attrs': study.best_trial.user_attrs,
                   'n_trials': len(study.trials)}, f, indent=2)

    all_trials = [{'number': t.number, 'value': t.value,
                   'params': t.params, 'user_attrs': t.user_attrs,
                   'state': str(t.state)} for t in study.trials]
    with open(f"{RESULTS_DIR}/optuna_all_trials.json", 'w') as f:
        json.dump(all_trials, f, indent=2, default=str)

    for plot_name, plot_fn in [
        ('history',     optuna.visualization.matplotlib.plot_optimization_history),
        ('importances', optuna.visualization.matplotlib.plot_param_importances),
        ('parallel',    optuna.visualization.matplotlib.plot_parallel_coordinate),
        ('slice',       optuna.visualization.matplotlib.plot_slice),
    ]:
        try:
            plot_fn(study)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/optuna_{plot_name}.png", dpi=120)
            plt.close()
        except Exception as e:
            print(f"  (skip {plot_name} plot: {e})")

    # -------------- FINAL RETRAIN --------------
    best = study.best_params
    best_attrs = study.best_trial.user_attrs
    modes        = min(best['modes'], N_STEPS // 2 + 1)
    width        = best_attrs['width']
    num_heads    = best_attrs['num_heads']
    depth        = best['depth']
    lr           = best['lr']
    batch_size   = best['batch_size']
    weight_decay = best['weight_decay']
    omega0       = best['omega0']
    dropout_rate = best['dropout_rate']

    print(f"\n--- Retraining best config at FULL budget ---")
    print(f"    modes={modes}, width={width}, heads={num_heads}, depth={depth}")
    print(f"    lr={lr:.2e}, bs={batch_size}, wd={weight_decay:.2e}, "
          f"omega0={omega0:.2f}, dropout={dropout_rate:.3f}")

    model = build_model(modes, width, depth, omega0, num_heads, dropout_rate,
                        seed=0)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))

    train_start = time_module.time()
    best_val, best_params_state, history, best_epoch, _ = train_model(
        model, opt_state, optimizer,
        _DATA['x_train'], _DATA['y_train'],
        _DATA['x_val'],   _DATA['y_val'],
        _DATA['x_test'],  _DATA['y_test'],
        batch_size=batch_size,
        epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE,
        trial=None, log_every=10,
        record_history=True, verbose=True,
    )
    training_time = time_module.time() - train_start
    print(f"  final training {training_time/60:.2f} min, best val {best_val:.6f}")

    if best_params_state is not None:
        nnx.update(model, best_params_state)

    model_config = {
        'modes': modes, 'width': width, 'depth': depth, 'omega0': omega0,
        'num_heads': num_heads, 'dropout_rate': dropout_rate,
        'n_steps': N_STEPS,
        'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay,
        'architecture': 'FNO1D_SIREN_Attn',
    }
    save_model(model, model_config, f"{RESULTS_DIR}/best_model_fno_attention.pkl")

    # -------------- PLOTS & TESTS --------------
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label="Train", color='blue')
    plt.plot(history['val'], label="Val", color='green', linestyle='--')
    plt.plot(history['test'], label="Test", color='orange', linestyle=':')
    plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.title('Final retrain — Brepols + Attention (Optuna-tuned)')
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=120)
    plt.close()

    time_axis = np.linspace(0, T_TOTAL, N_STEPS)
    ti = np.random.choice(_DATA['x_test'].shape[0], 3, replace=False)
    preds = model(_DATA['x_test'][ti])
    for i, idx in enumerate(ti):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].plot(time_axis, _DATA['x_test'][idx].flatten(), 'b-', linewidth=2)
        ax[0, 0].set_title(f'Sample {idx}: Strain'); ax[0, 0].grid(True, alpha=0.3)
        ax[0, 1].plot(time_axis, _DATA['y_test'][idx].flatten(), 'g-', linewidth=2, label='Truth')
        ax[0, 1].plot(time_axis, preds[i].flatten(), 'r--', linewidth=2, label='FNO')
        ax[0, 1].set_title('Stress'); ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)
        ax[1, 0].plot(_DATA['x_test'][idx].flatten(), _DATA['y_test'][idx].flatten(),
                      'g-', linewidth=2, label='Truth')
        ax[1, 0].plot(_DATA['x_test'][idx].flatten(), preds[i].flatten(),
                      'r--', linewidth=2, label='FNO')
        ax[1, 0].set_title('Hysteresis'); ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)
        err = np.abs(_DATA['y_test'][idx].flatten() - preds[i].flatten())
        ax[1, 1].plot(time_axis, err, 'orange', linewidth=2)
        ax[1, 1].set_title('Abs Error'); ax[1, 1].grid(True, alpha=0.3)
        plt.suptitle('Brepols Damage-Plasticity + Attention: FNO Prediction',
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/prediction_test_{i}.png", dpi=150)
        plt.close()

    test_xsinx_case(model, N_STEPS, T_TOTAL, RESULTS_DIR)

    # -------------- FULL TEST SUITE --------------
    test_t0 = time_module.time()
    comprehensive_statistical_testing(model, N_STEPS, RESULTS_DIR)
    test_discretization_sensitivity(
        model, RESULTS_DIR,
        resolutions=[50, 100, 150, 200, 250, 300, 350, 400,
                     450, 500, 800, 1000],
        num_test_samples=50, training_resolution=N_STEPS, seed=12345,
    )
    test_time = time_module.time() - test_t0

    save_timing_info(RESULTS_DIR, training_time, test_time, 300,
                     history, best_epoch)

    total = time_module.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL DONE — total {total/60:.1f} min. Results in '{RESULTS_DIR}/'")
    print("=" * 70)


if __name__ == "__main__":
    main()