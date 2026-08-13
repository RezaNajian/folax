"""
2D Plasticity FNO + SIREN + Causal Attention — TRAIN + TEST
==================================================================
This script picks up from a successful Optuna search (results below) and
runs the FINAL retrain + full test suite. 

Best Optuna trial #31 (val loss 0.000986):
    batch_size    = 64
    learning_rate = 0.0008247270656788965
    weight_decay  = 0.0004964179186952205
    modes         = 12
    width         = 48
    num_heads     = 4
    depth         = 5
    omega_0       = 9.368831675631814
    dropout_rate  = 8.223476842818183e-05


"""
import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import serialization
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pickle
import time as time_module
import traceback


# ==========================================================================
# 1. 2D PLANE STRAIN MATERIAL MODEL
# ==========================================================================
@jax.jit
def compute_stress_and_deviator(strain, plastic_strain, E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    eps_e_xx = strain[0] - plastic_strain[0]
    eps_e_yy = strain[1] - plastic_strain[1]
    eps_e_xy = strain[2] / 2.0 - plastic_strain[2] / 2.0
    eps_e_zz = plastic_strain[0] + plastic_strain[1]

    eps_vol = eps_e_xx + eps_e_yy + eps_e_zz

    sig_xx = lam * eps_vol + 2.0 * mu * eps_e_xx
    sig_yy = lam * eps_vol + 2.0 * mu * eps_e_yy
    sig_zz = lam * eps_vol + 2.0 * mu * eps_e_zz
    sig_xy = 2.0 * mu * eps_e_xy

    p = (sig_xx + sig_yy + sig_zz) / 3.0
    s_xx, s_yy, s_zz, s_xy = sig_xx - p, sig_yy - p, sig_zz - p, sig_xy

    return jnp.array([sig_xx, sig_yy, sig_xy]), s_xx, s_yy, s_zz, s_xy


@jax.jit
def von_mises_stress(s_xx, s_yy, s_zz, s_xy):
    return jnp.sqrt(1.5 * (s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * s_xy**2) + 1e-12)


@jax.jit
def material_step_2d(state, strain_new):
    plastic_strain_i, xi_i = state

    E, h1, h2, y0, nu = 3.0, 0.4, 10.0, 0.6, 0.3
    G = E / (2.0 * (1.0 + nu))

    stress_trial, s_xx_t, s_yy_t, s_zz_t, s_xy_t = compute_stress_and_deviator(
        strain_new, plastic_strain_i, E, nu
    )
    q_trial = von_mises_stress(s_xx_t, s_yy_t, s_zz_t, s_xy_t)
    sigma_y_i = y0 + h1 * (1.0 - jnp.exp(-h2 * xi_i))
    phi_trial = q_trial - sigma_y_i

    def elastic_update(_):
        return (plastic_strain_i, xi_i), stress_trial

    def plastic_update(_):
        q_safe = jnp.maximum(q_trial, 1e-8)
        n_xx = 1.5 * s_xx_t / q_safe
        n_yy = 1.5 * s_yy_t / q_safe
        n_xy = 1.5 * s_xy_t / q_safe

        def residual(dgamma):
            xi_new = xi_i + dgamma
            sigma_y_new = y0 + h1 * (1.0 - jnp.exp(-h2 * xi_new))
            return q_trial - 3.0 * G * dgamma - sigma_y_new

        tol, max_iter = 1e-10, 50

        def cond_fn(carry):
            _, f_val, it = carry
            return jnp.logical_and(jnp.abs(f_val) > tol, it < max_iter)

        def body_fn(carry):
            dgamma, _, it = carry
            f = residual(dgamma)
            df = jax.jacfwd(residual)(dgamma)
            dgamma_new = jnp.maximum(dgamma - f / df, 0.0)
            return (dgamma_new, residual(dgamma_new), it + 1)

        H_i = h1 * h2 * jnp.exp(-h2 * xi_i)
        dgamma_init = jnp.maximum(phi_trial / (3.0 * G + H_i), 0.0)
        init = (dgamma_init, residual(dgamma_init), 0)
        dgamma_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)

        plastic_strain_new = plastic_strain_i + dgamma_final * jnp.array(
            [n_xx, n_yy, 2.0 * n_xy]
        )
        xi_new = xi_i + dgamma_final
        stress_final, _, _, _, _ = compute_stress_and_deviator(
            strain_new, plastic_strain_new, E, nu
        )
        return (plastic_strain_new, xi_new), stress_final

    return jax.lax.cond(phi_trial > 0.0, plastic_update, elastic_update, operand=None)


def solve_full_path_2d(strain_history):
    initial_state = (jnp.zeros(3), 0.0)

    def scan_body(carry, strain):
        new_state, stress = material_step_2d(carry, strain)
        return new_state, stress

    _, stress_history = jax.lax.scan(scan_body, initial_state, strain_history)
    return stress_history


# ==========================================================================
# 2. DATA GENERATORS
# ==========================================================================
def _gp_path_starting_at_zero(num_steps, max_strain):
    t = np.linspace(0, 1, num_steps).reshape(-1, 1)
    length_scale = np.random.uniform(0.01, 0.1)
    K = np.exp(-cdist(t, t, "sqeuclidean") / (2 * length_scale**2)) + np.eye(num_steps) * 1e-6
    path = np.random.multivariate_normal(np.zeros(num_steps), K)
    path = path - path[0]
    m = np.max(np.abs(path))
    if m > 1e-6:
        path = (path / m) * np.random.uniform(0.1, max_strain)
    path[0] = 0.0
    return path


def generate_2d_gp_paths(num_samples, num_steps, max_strain=1.0, active_components=None):
    if active_components is None:
        active_components = [0, 1, 2]
    print(f"Generating {num_samples} 2D GP paths (active comps={active_components}, start=0)...")
    data = np.zeros((num_samples, num_steps, 3))
    for i in range(num_samples):
        for c in active_components:
            data[i, :, c] = _gp_path_starting_at_zero(num_steps, max_strain)
    return jnp.array(data)


def generate_2d_uniaxial_paths(num_samples, num_steps, max_strain=1.0):
    print(f"Generating {num_samples} 2D uniaxial paths (start=0)...")
    data = np.zeros((num_samples, num_steps, 3))
    for i in range(num_samples):
        c = i % 3
        data[i, :, c] = _gp_path_starting_at_zero(num_steps, max_strain)
    return jnp.array(data)


def generate_2d_biaxial_paths(num_samples, num_steps, max_strain=1.0):
    print(f"Generating {num_samples} 2D biaxial paths (start=0)...")
    data = np.zeros((num_samples, num_steps, 3))
    combos = [(0, 1), (0, 2), (1, 2)]
    for i in range(num_samples):
        c1, c2 = combos[i % 3]
        data[i, :, c1] = _gp_path_starting_at_zero(num_steps, max_strain)
        data[i, :, c2] = _gp_path_starting_at_zero(num_steps, max_strain)
    return jnp.array(data)


def generate_2d_random_zigzag_paths(num_samples, num_steps, max_strain=1.0,
                                     active_components=None):
    if active_components is None:
        active_components = [0, 1, 2]
    key_points = np.linspace(0, num_steps - 1, 7, dtype=int)
    data = np.zeros((num_samples, num_steps, 3))
    for i in range(num_samples):
        for c in active_components:
            checkpoints = np.random.uniform(-max_strain, max_strain, size=5)
            values = np.concatenate(([0.0], checkpoints, [0.0]))
            data[i, :, c] = np.interp(np.arange(num_steps), key_points, values)
    return jnp.array(data)


def generate_2d_sinusoidal_paths(num_samples, num_steps, max_strain=1.0, T_total=1.0,
                                  active_components=None):
    if active_components is None:
        active_components = [0, 1, 2]
    t = np.linspace(0, T_total, num_steps)
    data = np.zeros((num_samples, num_steps, 3))
    for i in range(num_samples):
        for c in active_components:
            freq = np.random.uniform(1.0, 5.0)
            amp = np.random.uniform(0.3, max_strain)
            phase = np.random.uniform(0, 2 * np.pi)
            sig = amp * 0.5 * (1.0 - np.cos(2 * np.pi * freq * t + phase))
            sig = sig - sig[0]
            data[i, :, c] = sig
    return jnp.array(data)


# ==========================================================================
# 3. MODEL: FNO + SIREN + CAUSAL ATTENTION
# ==========================================================================

class SirenLinear(nnx.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0, rngs=None):
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega0 = omega0
        key = rngs.params()
        if is_first:
            limit = 1.0 / in_features
        else:
            limit = jnp.sqrt(6.0 / in_features) / omega0
        w_init = jax.random.uniform(
            key, (in_features, out_features), minval=-limit, maxval=limit
        )
        self.w = nnx.Param(w_init)
        self.b = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x):
        x = x @ self.w.value + self.b.value
        if self.is_first:
            x = self.omega0 * x
        return x


class SpectralConv1D(nnx.Module):
    def __init__(self, in_channels, out_channels, modes, rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (in_channels, out_channels, modes, 2),
                minval=-scale,
                maxval=scale,
            )
        )

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
        return jnp.fft.irfft(out_ft_padded, n=T, axis=1)


class CausalMultiHeadSelfAttention(nnx.Module):
    """Multi-head self-attention with strict causal mask along time axis."""
    def __init__(self, dim, num_heads=4, dropout_rate=0.1, rngs=None):
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )
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
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn_logits = jnp.where(
            causal_mask[None, None, :, :], attn_logits,
            jnp.finfo(attn_logits.dtype).min
        )

        attn = jax.nn.softmax(attn_logits, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not training)

        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)

        out = self.proj(out)
        out = self.proj_dropout(out, deterministic=not training)
        return out


class FNOAttnBlock2D(nnx.Module):
    """Pre-norm causal attention (residual) + FNO sub-block."""
    def __init__(self, width, modes, omega0=30.0, num_heads=4,
                 dropout_rate=0.1, rngs=None):
        self.spectral = SpectralConv1D(width, width, modes, rngs)
        self.w = SirenLinear(width, width, is_first=False, omega0=omega0, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(width, rngs=rngs)
        self.attn = CausalMultiHeadSelfAttention(
            dim=width, num_heads=num_heads, dropout_rate=dropout_rate, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, training=False):
        x = x + self.attn(self.attn_norm(x), training=training)
        x_spec = self.spectral(x)
        x_loc = self.w(x)
        x = jnp.sin(x_spec + x_loc)
        x = self.dropout(x, deterministic=not training)
        return x


class FNO1D_2DPlasticity_Attn(nnx.Module):
    def __init__(self, modes, width, depth, rngs, padding_frac=0.1,
                 omega0=30.0, num_heads=4, dropout_rate=0.1):
        self.width = width
        self.modes = modes
        self.padding_frac = padding_frac
        lifting_in_dim = 3

        self.fc0 = SirenLinear(lifting_in_dim, width, is_first=True,
                               omega0=omega0, rngs=rngs)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(
                FNOAttnBlock2D(width=width, modes=modes, omega0=omega0,
                               num_heads=num_heads, dropout_rate=dropout_rate,
                               rngs=rngs)
            )

        self.fc1 = SirenLinear(width, 128, is_first=False, omega0=omega0, rngs=rngs)
        self.fc2_normal = nnx.Linear(128, 2, rngs=rngs)
        self.fc2_shear = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x, training=False):
        B, T, _ = x.shape
        x = jnp.sin(self.fc0(x))

        n_pad = int(T * self.padding_frac)
        if n_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)))

        for block in self.blocks:
            x = block(x, training=training)

        if n_pad > 0:
            x = x[:, :T, :]

        x = jnp.sin(self.fc1(x))
        x_n = self.fc2_normal(x)
        x_s = self.fc2_shear(x)
        x = jnp.concatenate([x_n, x_s], axis=-1)

        x = x - x[:, 0:1, :]
        return x


# ==========================================================================
# 4. NORMALIZER
# ==========================================================================
class DataNormalizer2D:
    def __init__(self, strains, stresses):
        self.strain_mean = strains.mean(axis=(0, 1))
        self.strain_std = strains.std(axis=(0, 1)) + 1e-8
        self.stress_mean = stresses.mean(axis=(0, 1))
        self.stress_std = stresses.std(axis=(0, 1)) + 1e-8

    def norm_strain(self, x):
        return (x - self.strain_mean) / self.strain_std

    def norm_stress(self, x):
        return (x - self.stress_mean) / self.stress_std

    def denorm_stress(self, x):
        return x * self.stress_std + self.stress_mean


# ==========================================================================
# 5. UTILITIES
# ==========================================================================
def relative_l2_error(pred, truth, eps=1e-10):
    num = jnp.sqrt(jnp.mean((pred - truth) ** 2))
    den = jnp.sqrt(jnp.mean(truth ** 2)) + eps
    return float(num / den)


def save_model_safe(model, opt_state, model_config, history, filepath, normalizer=None,
                    weights_only=False):
    """
    Robust save: pickle a numpy-converted weights dict + config + history.

    If ``weights_only`` is True, opt_state is omitted (used by the
    in-training periodic checkpoint where we don't care about resuming
    the optimizer state).
    """
    print(f"\n   Saving model to {filepath} ...")
    try:
        params = nnx.state(model, nnx.Param)
        params_np = jax.tree_util.tree_map(
            lambda x: np.asarray(x) if hasattr(x, "shape") else x, params
        )
        opt_bytes = None
        if not weights_only and opt_state is not None:
            try:
                opt_bytes = serialization.to_bytes(opt_state)
            except (TypeError, ValueError) as e:
                print(f"   Warning: could not serialize optimizer state ({e}); "
                      f"saving weights only.")
        checkpoint = {
            "params": params_np,
            "opt_state_bytes": opt_bytes,
            "model_config": model_config,
            "history": history,
            "normalizer": normalizer,
        }
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"   Saved ({os.path.getsize(filepath)/1e6:.2f} MB).")
        return True
    except Exception as e:
        print(f"   Save failed: {e}")
        traceback.print_exc()
        return False


def predict_denorm(model, strain_raw, normalizer, training=False):
    if normalizer is None:
        return np.asarray(model(jnp.asarray(strain_raw), training=training))
    strain_norm = normalizer.norm_strain(strain_raw)
    pred_norm = model(jnp.asarray(strain_norm), training=training)
    return normalizer.denorm_stress(np.asarray(pred_norm))


# ==========================================================================
# 6. TRAINING ROUTINE
# ==========================================================================
def train_one_model(*,
    x_train_in, y_train_in,
    x_val_in, y_val_in,
    x_test_in, y_test_in,
    modes, width, depth, omega_0, num_heads, dropout_rate,
    learning_rate, weight_decay, batch_size,
    epochs, patience,
    seed=0,
    checkpoint_path=None,
    checkpoint_every=200,
    model_config_for_ckpt=None,
    normalizer_for_ckpt=None,
    verbose=True,
):
    model = FNO1D_2DPlasticity_Attn(
        modes=modes, width=width, depth=depth, rngs=nnx.Rngs(seed),
        omega0=omega_0,
        num_heads=num_heads, dropout_rate=dropout_rate,
    )
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))

    @nnx.jit
    def train_step(model, opt_state, batch_x, batch_y):
        def loss_fn(model):
            pred = model(batch_x, training=True)
            weights = jnp.array([1.0, 1.0, 3.0])
            return jnp.mean((pred - batch_y) ** 2 * weights[None, None, :])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        params = nnx.state(model, nnx.Param)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return loss, new_opt_state

    @nnx.jit
    def eval_step(model, batch_x, batch_y):
        pred = model(batch_x, training=False)
        return jnp.mean((pred - batch_y) ** 2)

    n_train_b = max(1, x_train_in.shape[0] // batch_size)
    n_val_b = max(1, math.ceil(x_val_in.shape[0] / batch_size))
    n_test_b = max(1, math.ceil(x_test_in.shape[0] / batch_size))

    history = {"train": [], "val": [], "test": []}
    best_val_loss = float("inf")
    best_params = None
    patience_counter = 0
    epochs_done = 0

    for epoch in range(epochs):
        epochs_done = epoch + 1
        perm = np.random.permutation(x_train_in.shape[0])
        x_s, y_s = x_train_in[perm], y_train_in[perm]

        tl = 0.0
        for i in range(n_train_b):
            bx = x_s[i * batch_size:(i + 1) * batch_size]
            by = y_s[i * batch_size:(i + 1) * batch_size]
            loss, opt_state = train_step(model, opt_state, bx, by)
            tl += loss
        tl /= n_train_b

        vl = 0.0
        for i in range(n_val_b):
            bx = x_val_in[i * batch_size:(i + 1) * batch_size]
            by = y_val_in[i * batch_size:(i + 1) * batch_size]
            vl += eval_step(model, bx, by)
        vl /= n_val_b

        tel = 0.0
        for i in range(n_test_b):
            bx = x_test_in[i * batch_size:(i + 1) * batch_size]
            by = y_test_in[i * batch_size:(i + 1) * batch_size]
            tel += eval_step(model, bx, by)
        tel /= n_test_b

        if not (np.isfinite(float(tl)) and np.isfinite(float(vl))):
            print(f"  Non-finite loss at epoch {epoch}; stopping.")
            break

        if vl < best_val_loss:
            best_val_loss = float(vl)
            patience_counter = 0
            best_params = nnx.state(model)


            if checkpoint_path is not None and model_config_for_ckpt is not None:
                save_model_safe(
                    model, None, model_config_for_ckpt,
                    history, checkpoint_path, normalizer_for_ckpt,
                    weights_only=True,
                )
        else:
            patience_counter += 1

        history["train"].append(float(tl))
        history["val"].append(float(vl))
        history["test"].append(float(tel))

        if verbose and (epoch % 10 == 0):
            print(f"  Epoch {epoch}: Train={tl:.6f} | Val={vl:.6f} | Test={tel:.6f}"
                  f" | best_val={best_val_loss:.6f}")

        if (checkpoint_path is not None
                and model_config_for_ckpt is not None
                and (epoch + 1) % checkpoint_every == 0):
            ckpt_periodic = checkpoint_path.replace(".pkl", "_periodic.pkl")
            save_model_safe(
                model, None, model_config_for_ckpt,
                history, ckpt_periodic, normalizer_for_ckpt,
                weights_only=True,
            )

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}.")
            break

    if best_params is not None:
        nnx.update(model, best_params)

    return model, opt_state, history, best_val_loss, best_params, epochs_done


# ==========================================================================
# 7. COMPREHENSIVE STATISTICAL TESTING
# ==========================================================================
def comprehensive_statistical_testing_2d(model, batch_solver, num_steps, results_dir,
                                         normalizer=None):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE 2D STATISTICAL TESTING (relative L2)")
    if normalizer is not None:
        print("  (model trained with normalization; predictions denormalized for error)")
    print("=" * 80)

    T_TOTAL = 1.0
    n_samples = 100
    chunk_size = 20
    samples_per_comp = [33, 33, 34]
    biaxial_combos = [(0, 1), (0, 2), (1, 2)]
    biaxial_samples = [33, 33, 34]
    errors = {k: [] for k in [
        "uniaxial_gp", "uniaxial_sino", "uniaxial_random",
        "biaxial_gp", "biaxial_sino", "biaxial_random",
        "multiaxial_gp", "multiaxial_sino", "multiaxial_random",
    ]}

    def _run_block(inputs, key):
        truth_list = []
        for i in range(0, inputs.shape[0], chunk_size):
            truth_list.append(batch_solver(inputs[i : i + chunk_size]))
        truth = np.asarray(jnp.concatenate(truth_list, axis=0))
        pred = predict_denorm(model, np.asarray(inputs), normalizer)
        for i in range(inputs.shape[0]):
            errors[key].append(relative_l2_error(pred[i], truth[i]))

    print("\n[1/9] Uniaxial GP ...")
    for c, n in enumerate(samples_per_comp):
        _run_block(generate_2d_gp_paths(n, num_steps, 1.0, [c]), "uniaxial_gp")
    print(f"  Mean rel-L2 = {np.mean(errors['uniaxial_gp']):.6f}, "
          f"Std = {np.std(errors['uniaxial_gp']):.6f}")

    print("\n[2/9] Uniaxial Sinusoidal ...")
    for c, n in enumerate(samples_per_comp):
        _run_block(
            generate_2d_sinusoidal_paths(n, num_steps, 1.0, T_TOTAL, [c]),
            "uniaxial_sino",
        )
    print(f"  Mean rel-L2 = {np.mean(errors['uniaxial_sino']):.6f}")

    print("\n[3/9] Uniaxial Random ...")
    for c, n in enumerate(samples_per_comp):
        _run_block(
            generate_2d_random_zigzag_paths(n, num_steps, 1.0, [c]),
            "uniaxial_random",
        )
    print(f"  Mean rel-L2 = {np.mean(errors['uniaxial_random']):.6f}")

    print("\n[4/9] Biaxial GP ...")
    for (c1, c2), n in zip(biaxial_combos, biaxial_samples):
        _run_block(generate_2d_gp_paths(n, num_steps, 1.0, [c1, c2]), "biaxial_gp")
    print(f"  Mean rel-L2 = {np.mean(errors['biaxial_gp']):.6f}")

    print("\n[5/9] Biaxial Sinusoidal ...")
    for (c1, c2), n in zip(biaxial_combos, biaxial_samples):
        _run_block(
            generate_2d_sinusoidal_paths(n, num_steps, 1.0, T_TOTAL, [c1, c2]),
            "biaxial_sino",
        )
    print(f"  Mean rel-L2 = {np.mean(errors['biaxial_sino']):.6f}")

    print("\n[6/9] Biaxial Random ...")
    for (c1, c2), n in zip(biaxial_combos, biaxial_samples):
        _run_block(
            generate_2d_random_zigzag_paths(n, num_steps, 1.0, [c1, c2]),
            "biaxial_random",
        )
    print(f"  Mean rel-L2 = {np.mean(errors['biaxial_random']):.6f}")

    print("\n[7/9] Multiaxial GP ...")
    _run_block(generate_2d_gp_paths(n_samples, num_steps, 1.0), "multiaxial_gp")
    print(f"  Mean rel-L2 = {np.mean(errors['multiaxial_gp']):.6f}")

    print("\n[8/9] Multiaxial Sinusoidal ...")
    _run_block(
        generate_2d_sinusoidal_paths(n_samples, num_steps, 1.0, T_TOTAL),
        "multiaxial_sino",
    )
    print(f"  Mean rel-L2 = {np.mean(errors['multiaxial_sino']):.6f}")

    print("\n[9/9] Multiaxial Random ...")
    _run_block(generate_2d_random_zigzag_paths(n_samples, num_steps, 1.0), "multiaxial_random")
    print(f"  Mean rel-L2 = {np.mean(errors['multiaxial_random']):.6f}")

    print("\n[Plotting] ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cats = ["Uniaxial", "Biaxial", "Multiaxial"]
    keys_grouped = [
        ["uniaxial_gp", "uniaxial_sino", "uniaxial_random"],
        ["biaxial_gp", "biaxial_sino", "biaxial_random"],
        ["multiaxial_gp", "multiaxial_sino", "multiaxial_random"],
    ]
    for ax, cat, ks in zip(axes, cats, keys_grouped):
        bp = ax.boxplot(
            [errors[k] for k in ks],
            labels=["GP", "Sinusoidal", "Random"],
            patch_artist=True, showmeans=True,
        )
        for patch, color in zip(bp["boxes"], ["lightblue", "lightcoral", "lightgreen"]):
            patch.set_facecolor(color)
        ax.set_ylabel("Relative L2 Error")
        ax.set_title(f"{cat} Loading (n=100 each)", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("2D Plasticity FNO + Attention: Relative L2 Error Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/2d_statistical_boxplots_by_category.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [
        "Uni-GP", "Uni-Sino", "Uni-Random",
        "Bi-GP", "Bi-Sino", "Bi-Random",
        "Multi-GP", "Multi-Sino", "Multi-Random",
    ]
    keys_flat = [
        "uniaxial_gp", "uniaxial_sino", "uniaxial_random",
        "biaxial_gp", "biaxial_sino", "biaxial_random",
        "multiaxial_gp", "multiaxial_sino", "multiaxial_random",
    ]
    means = [np.mean(errors[k]) for k in keys_flat]
    stds = [np.std(errors[k]) for k in keys_flat]
    colors = ["blue", "red", "green"] * 3
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, alpha=0.7,
           color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Mean Relative L2 Error")
    ax.set_title("2D FNO + Attention: Mean Relative L2 Error (±1σ) over 9 scenarios",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/2d_statistical_comparison_bar.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    with open(f"{results_dir}/2d_statistical_summary.txt", "w") as f:
        f.write("2D Plasticity FNO + Attention: comprehensive relative L2 error testing\n")
        f.write("=" * 70 + "\n\n")
        for k in keys_flat:
            data = errors[k]
            f.write(f"{k}\n")
            f.write(f"  Mean   = {np.mean(data):.8f}\n")
            f.write(f"  Std    = {np.std(data):.8f}\n")
            f.write(f"  Median = {np.median(data):.8f}\n")
            f.write(f"  P95    = {np.percentile(data, 95):.8f}\n\n")

    print("Saved statistical plots and summary.")
    return errors


# ==========================================================================
# 8. DISCRETIZATION SENSITIVITY STUDY
# ==========================================================================
def generate_canonical_zigzag_2d(num_uni, num_bi, num_multi, num_key_points=7,
                                 max_strain=1.0, seed=12345):
    rng = np.random.default_rng(seed)
    key_points = np.linspace(0, 1, num_key_points)
    specs = []

    for i in range(num_uni):
        active = [i % 3]
        kv = np.zeros((num_key_points, 3))
        for c in active:
            interior = rng.uniform(-max_strain, max_strain, size=num_key_points - 2)
            kv[:, c] = np.concatenate(([0.0], interior, [0.0]))
        specs.append({"key_points": key_points, "key_values": kv, "active": active,
                      "type": "uniaxial"})

    combos = [(0, 1), (0, 2), (1, 2)]
    for i in range(num_bi):
        active = list(combos[i % 3])
        kv = np.zeros((num_key_points, 3))
        for c in active:
            interior = rng.uniform(-max_strain, max_strain, size=num_key_points - 2)
            kv[:, c] = np.concatenate(([0.0], interior, [0.0]))
        specs.append({"key_points": key_points, "key_values": kv, "active": active,
                      "type": "biaxial"})

    for i in range(num_multi):
        active = [0, 1, 2]
        kv = np.zeros((num_key_points, 3))
        for c in active:
            interior = rng.uniform(-max_strain, max_strain, size=num_key_points - 2)
            kv[:, c] = np.concatenate(([0.0], interior, [0.0]))
        specs.append({"key_points": key_points, "key_values": kv, "active": active,
                      "type": "multiaxial"})

    return specs


def resample_spec_to_resolution(spec, num_steps):
    step_pos = np.linspace(0, 1, num_steps)
    out = np.zeros((num_steps, 3))
    for c in range(3):
        out[:, c] = np.interp(step_pos, spec["key_points"], spec["key_values"][:, c])
    out[0, :] = 0.0
    return out


def test_discretization_sensitivity_2d(model, results_dir,
                                        resolutions=(50, 100, 150, 200, 250, 300, 350, 400,
                                                     450, 500, 800, 1000),
                                        training_resolution=50, seed=12345,
                                        normalizer=None):
    print("\n" + "=" * 80)
    print("DISCRETIZATION SENSITIVITY TESTING (2D)")
    if normalizer is not None:
        print("  (model trained with normalization; predictions denormalized for error)")
    print("=" * 80)
    print(f"Resolutions      : {list(resolutions)}")
    print(f"Training resolution : {training_resolution}")
    print(f"Test bank        : 33 uniaxial + 33 biaxial + 34 multiaxial = 100")

    specs = generate_canonical_zigzag_2d(33, 33, 34, num_key_points=7,
                                         max_strain=1.0, seed=seed)
    spec_types = np.array([s["type"] for s in specs])
    n_specs = len(specs)

    batch_solver = jax.vmap(solve_full_path_2d)

    overall_mean, overall_std = [], []
    cat_means = {"uniaxial": [], "biaxial": [], "multiaxial": []}
    cat_stds = {"uniaxial": [], "biaxial": [], "multiaxial": []}
    per_resolution_errors = {}

    for r_idx, num_steps in enumerate(resolutions):
        print(f"\n--- Resolution {num_steps} ({r_idx+1}/{len(resolutions)}) ---")

        strain_array = np.zeros((n_specs, num_steps, 3))
        for i, spec in enumerate(specs):
            strain_array[i] = resample_spec_to_resolution(spec, num_steps)
        strain_jax = jnp.array(strain_array)

        chunk = 25
        truth_list = []
        for i in range(0, n_specs, chunk):
            truth_list.append(batch_solver(strain_jax[i:i + chunk]))
        truth = np.asarray(jnp.concatenate(truth_list, axis=0))

        pred = predict_denorm(model, strain_array, normalizer)

        sample_errors = np.array(
            [relative_l2_error(pred[i], truth[i]) for i in range(n_specs)]
        )
        per_resolution_errors[num_steps] = sample_errors

        overall_mean.append(float(np.mean(sample_errors)))
        overall_std.append(float(np.std(sample_errors)))

        for cat in ["uniaxial", "biaxial", "multiaxial"]:
            mask = spec_types == cat
            cat_means[cat].append(float(np.mean(sample_errors[mask])))
            cat_stds[cat].append(float(np.std(sample_errors[mask])))

        print(f"  Overall mean rel-L2 = {overall_mean[-1]*100:.3f}% "
              f"(std {overall_std[-1]*100:.3f}%)")

    overall_mean = np.array(overall_mean)
    overall_std = np.array(overall_std)
    res_arr = np.array(resolutions)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res_arr, overall_mean * 100, "o-", color="black",
            linewidth=2.5, markersize=7, label="Overall (n=100)")
    ax.fill_between(res_arr,
                    (overall_mean - overall_std) * 100,
                    (overall_mean + overall_std) * 100,
                    alpha=0.15, color="black", label="±1σ overall")

    cat_colors = {"uniaxial": "tab:blue", "biaxial": "tab:orange", "multiaxial": "tab:green"}
    cat_n = {"uniaxial": 33, "biaxial": 33, "multiaxial": 34}
    for cat in ["uniaxial", "biaxial", "multiaxial"]:
        ax.plot(res_arr, np.array(cat_means[cat]) * 100, "s--",
                color=cat_colors[cat], linewidth=1.6, markersize=5, alpha=0.9,
                label=f"{cat.capitalize()} (n={cat_n[cat]})")

    if training_resolution in resolutions:
        ax.axvline(training_resolution, color="red", linestyle=":", linewidth=2,
                   label=f"Training resolution = {training_resolution}")

    ax.set_xlabel("Number of time steps", fontsize=12)
    ax.set_ylabel("Mean Relative L2 Error (%)", fontsize=12)
    ax.set_title("2D FNO + Attention Discretization Sensitivity\n"
                 "(100 canonical zig-zag paths)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/2d_discretization_mean_relL2_vs_steps.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([per_resolution_errors[r] for r in resolutions],
                    labels=[str(r) for r in resolutions],
                    patch_artist=True, showmeans=True)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(resolutions)))
    for patch, c in zip(bp["boxes"], cmap):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    if training_resolution in resolutions:
        ax.axvline(list(resolutions).index(training_resolution) + 1,
                   color="red", linestyle=":", linewidth=2,
                   label=f"Training resolution = {training_resolution}")
        ax.legend()
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("2D FNO + Attention: Per-sample relative L2 errors per resolution",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/2d_discretization_boxplot.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    with open(f"{results_dir}/2d_discretization_summary.txt", "w") as f:
        f.write("2D Plasticity FNO + Attention — discretization sensitivity\n")
        f.write("=" * 70 + "\n")
        f.write(f"Training resolution : {training_resolution}\n")
        f.write(f"Test bank           : 33 uniaxial + 33 biaxial + 34 multiaxial = 100\n")
        f.write(f"Metric              : relative L2 error (per sample)\n\n")
        f.write(
            f"{'Steps':<8}{'Overall mean (%)':<20}{'Std (%)':<12}"
            f"{'Uni mean (%)':<15}{'Bi mean (%)':<15}{'Multi mean (%)':<15}\n"
        )
        f.write("-" * 85 + "\n")
        for i, r in enumerate(resolutions):
            f.write(
                f"{r:<8}{overall_mean[i]*100:<20.3f}{overall_std[i]*100:<12.3f}"
                f"{cat_means['uniaxial'][i]*100:<15.3f}"
                f"{cat_means['biaxial'][i]*100:<15.3f}"
                f"{cat_means['multiaxial'][i]*100:<15.3f}\n"
            )
    print(f"\nSaved plots and summary to '{results_dir}'.")
    return {
        "resolutions": list(resolutions),
        "overall_mean": overall_mean.tolist(),
        "overall_std": overall_std.tolist(),
        "cat_means": cat_means,
        "cat_stds": cat_stds,
        "per_resolution_errors": {r: per_resolution_errors[r].tolist() for r in resolutions},
    }


# ==========================================================================
# 9. MAIN
# ==========================================================================
def main():
    # ---------- BEST OPTUNA HYPERPARAMETERS (HARDCODED) -------------------
    # From completed Optuna study, trial #31, best val loss 0.000986.
    BEST = {
        "batch_size":    64,
        "learning_rate": 0.0008247270656788965,
        "weight_decay":  0.0004964179186952205,
        "modes":         12,
        "width":         48,
        "num_heads":     4,
        "depth":         5,
        "omega_0":       9.368831675631814,
        "dropout_rate":  8.223476842818183e-05,
    }

    # ---------- CONFIG ----------------------------------------------------
    N_STEPS = 50
    N_GP_MULTIAXIAL = 10000
    N_UNIAXIAL = 2000
    N_BIAXIAL = 2000
    N_TEST = 2000

    FINAL_EPOCHS    = 10000
    FINAL_PATIENCE  = 3000
    CHECKPOINT_EVERY = 200   # epochs between periodic snapshot saves
    USE_NORMALIZATION = True

    RESULTS_DIR = "Material_operator_2D_elastoplasticity"
    os.makedirs(RESULTS_DIR, exist_ok=True)


    with open(f"{RESULTS_DIR}/best_hyperparameters.txt", "w") as f:
        f.write("Best hyperparameters from completed Optuna study\n")
        f.write("(loaded from trial #31, best val loss 0.000986)\n")
        f.write("=" * 60 + "\n")
        for k, v in BEST.items():
            f.write(f"  {k} = {v}\n")

    np.random.seed(0)

    # ---------- DATA ------------------------------------------------------
    print("=" * 60)
    print("GENERATING TRAINING DATA (every path starts at 0)")
    print("=" * 60)

    train_x_multi = generate_2d_gp_paths(N_GP_MULTIAXIAL, N_STEPS, max_strain=1.0)
    train_x_uni = generate_2d_uniaxial_paths(N_UNIAXIAL, N_STEPS, max_strain=1.0)
    train_x_bi = generate_2d_biaxial_paths(N_BIAXIAL, N_STEPS, max_strain=1.0)
    full_train_x = jnp.concatenate([train_x_multi, train_x_uni, train_x_bi], axis=0)

    print(f"Total train samples: {full_train_x.shape[0]}")

    test_x = generate_2d_random_zigzag_paths(N_TEST, N_STEPS, max_strain=1.0)

    print("\n--- Solving ground truth ---")
    batch_solver = jax.vmap(solve_full_path_2d)

    chunk = 500
    y_full = []
    for i in range(0, full_train_x.shape[0], chunk):
        y_full.append(batch_solver(full_train_x[i:i + chunk]))
        print(f"  Solved {min(i+chunk, full_train_x.shape[0])}/{full_train_x.shape[0]}")
    full_train_y = jnp.concatenate(y_full, axis=0)

    test_y = []
    for i in range(0, test_x.shape[0], chunk):
        test_y.append(batch_solver(test_x[i:i + chunk]))
    test_y = jnp.concatenate(test_y, axis=0)

    indices = np.arange(full_train_x.shape[0])
    np.random.shuffle(indices)
    split_idx = int(0.9 * full_train_x.shape[0])
    x_train = full_train_x[indices[:split_idx]]
    y_train = full_train_y[indices[:split_idx]]
    x_val = full_train_x[indices[split_idx:]]
    y_val = full_train_y[indices[split_idx:]]
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {test_x.shape[0]}")

    # ---------- NORMALIZATION ---------------------------------------------
    if USE_NORMALIZATION:
        print("\n--- Fitting per-component normalizer on training data ---")
        normalizer = DataNormalizer2D(np.asarray(x_train), np.asarray(y_train))
        print(f"  Strain mean (per comp): {normalizer.strain_mean}")
        print(f"  Strain std  (per comp): {normalizer.strain_std}")
        print(f"  Stress mean (per comp): {normalizer.stress_mean}")
        print(f"  Stress std  (per comp): {normalizer.stress_std}")

        x_train_in = jnp.asarray(normalizer.norm_strain(np.asarray(x_train)))
        y_train_in = jnp.asarray(normalizer.norm_stress(np.asarray(y_train)))
        x_val_in = jnp.asarray(normalizer.norm_strain(np.asarray(x_val)))
        y_val_in = jnp.asarray(normalizer.norm_stress(np.asarray(y_val)))
        x_test_in = jnp.asarray(normalizer.norm_strain(np.asarray(test_x)))
        y_test_in = jnp.asarray(normalizer.norm_stress(np.asarray(test_y)))
    else:
        normalizer = None
        x_train_in, y_train_in = x_train, y_train
        x_val_in, y_val_in = x_val, y_val
        x_test_in, y_test_in = test_x, test_y

    T = x_train_in.shape[1]
    BEST_MODES = min(int(BEST["modes"]), T // 2 + 1)

    model_config = {
        "modes": BEST_MODES,
        "width": int(BEST["width"]),
        "depth": int(BEST["depth"]),
        "omega0": float(BEST["omega_0"]),
        "num_heads": int(BEST["num_heads"]),
        "dropout_rate": float(BEST["dropout_rate"]),
        "n_steps": N_STEPS,
        "use_normalization": USE_NORMALIZATION,
        "learning_rate": float(BEST["learning_rate"]),
        "weight_decay": float(BEST["weight_decay"]),
        "batch_size": int(BEST["batch_size"]),
        "architecture": "FNO1D_2DPlasticity_Attn",
    }

    # ---------- FINAL TRAIN -----------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL TRAINING WITH BEST OPTUNA PARAMETERS")
    print("=" * 80)
    print(f"  modes={BEST_MODES}, width={BEST['width']}, heads={BEST['num_heads']}, "
          f"depth={BEST['depth']}, omega0={BEST['omega_0']:.4f}")
    print(f"  lr={BEST['learning_rate']:.4e}, wd={BEST['weight_decay']:.4e}, "
          f"bs={BEST['batch_size']}, dropout={BEST['dropout_rate']:.6f}")
    print(f"  epochs={FINAL_EPOCHS}, patience={FINAL_PATIENCE}")
    print(f"  best-weights checkpoint -> {RESULTS_DIR}/best_model_checkpoint.pkl")
    print(f"  periodic snapshot every {CHECKPOINT_EVERY} epochs "
          f"-> {RESULTS_DIR}/best_model_checkpoint_periodic.pkl")

    t_final_0 = time_module.time()
    try:
        model, opt_state, history, best_val_loss, _, epochs_done = train_one_model(
            x_train_in=x_train_in, y_train_in=y_train_in,
            x_val_in=x_val_in,     y_val_in=y_val_in,
            x_test_in=x_test_in,   y_test_in=y_test_in,
            modes=BEST_MODES,
            width=int(BEST["width"]),
            depth=int(BEST["depth"]),
            omega_0=float(BEST["omega_0"]),
            num_heads=int(BEST["num_heads"]),
            dropout_rate=float(BEST["dropout_rate"]),
            learning_rate=float(BEST["learning_rate"]),
            weight_decay=float(BEST["weight_decay"]),
            batch_size=int(BEST["batch_size"]),
            epochs=FINAL_EPOCHS,
            patience=FINAL_PATIENCE,
            seed=0,
            checkpoint_path=f"{RESULTS_DIR}/best_model_checkpoint.pkl",
            checkpoint_every=CHECKPOINT_EVERY,
            model_config_for_ckpt=model_config,
            normalizer_for_ckpt=normalizer,
            verbose=True,
        )
    except Exception as e:
        print(f"\n!!! Training crashed: {e}")
        traceback.print_exc()
        print("    Checkpoint files in the results dir contain the best weights "
              "found so far. Cannot continue to testing in this session.")
        return

    final_time = time_module.time() - t_final_0
    print(f"\nFinal training time: {final_time/60:.2f} min "
          f"({epochs_done} epochs, best val loss: {best_val_loss:.6f})")

    # ---------- LOSS CURVE ------------------------------------------------
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history["train"], label="Train")
        plt.plot(history["val"], label="Val")
        plt.plot(history["test"], label="Test")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted MSE")
        plt.title("2D Plasticity FNO + Attention Loss Curve\n"
                  "(Optuna-best hyperparameters, final retrain)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Loss curve plot failed: {e}")

    # ---------- SAVE FINAL MODEL ------------------------------------------
    save_model_safe(
        model, opt_state, model_config, history,
        f"{RESULTS_DIR}/best_model.pkl", normalizer=normalizer,
    )

    save_model_safe(
        model, None, model_config, history,
        f"{RESULTS_DIR}/best_model_weights_only.pkl", normalizer=normalizer,
        weights_only=True,
    )

    # ---------- COMPREHENSIVE STATISTICAL TESTING -------------------------
    try:
        comprehensive_statistical_testing_2d(
            model=model,
            batch_solver=batch_solver,
            num_steps=N_STEPS,
            results_dir=RESULTS_DIR,
            normalizer=normalizer,
        )
    except Exception as e:
        print(f"\nStatistical testing crashed: {e}")
        traceback.print_exc()
        print("  Continuing to discretization study...")

    # ---------- DISCRETIZATION SENSITIVITY STUDY --------------------------
    try:
        test_discretization_sensitivity_2d(
            model=model,
            results_dir=RESULTS_DIR,
            resolutions=(50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000),
            training_resolution=N_STEPS,
            seed=12345,
            normalizer=normalizer,
        )
    except Exception as e:
        print(f"\nDiscretization sensitivity study crashed: {e}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"All results saved to '{RESULTS_DIR}/'")
    print("Key files:")
    print("  - best_hyperparameters.txt  (the Optuna-tuned config)")
    print("  - best_model.pkl            (final model + optimizer state)")
    print("  - best_model_weights_only.pkl  (defensive backup)")
    print("  - best_model_checkpoint.pkl    (best-so-far during training)")
    print("  - loss_curve.png")
    print("  - 2d_statistical_boxplots_by_category.png")
    print("  - 2d_statistical_comparison_bar.png")
    print("  - 2d_statistical_summary.txt")
    print("  - 2d_discretization_mean_relL2_vs_steps.png")
    print("  - 2d_discretization_boxplot.png")
    print("  - 2d_discretization_summary.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()