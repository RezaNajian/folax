"""
FNO + SIREN + Causal Self-Attention Model for 1D elastoplasticity.
With optuna hyperparameter tuning 

"""

import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import itertools
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pickle
import time as time_module
from flax import serialization
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ==========================================
# 1. USER'S MATERIAL MODEL (JAX IMPLEMENTATION)
# ==========================================
@jax.jit
def residual_vec(theta, ep, epP_i, xiP_i, prop):
    epP, xiP = theta
    E, h1, h2, y0 = prop
    str0 = E * (ep - epP)
    sgn = jnp.where(str0 == 0.0, 1.0, jnp.sign(str0))
    r0 = epP - epP_i - (xiP - xiP_i) * sgn
    r1 = jnp.abs(str0) - (y0 + h1 * (1.0 - jnp.exp(-h2 * xiP)))
    return jnp.stack([r0, r1])

@jax.jit
def material_step(state, ep_new):
    epP_i, xiP_i = state
    E, h1, h2, y0 = 3.0, 0.4, 10.0, 0.6
    prop = (E, h1, h2, y0)

    sigT = E * (ep_new - epP_i)
    phi = jnp.abs(sigT) - (y0 + h1 * (1.0 - jnp.exp(-h2 * xiP_i)))

    def elastic_update(_):
        return (epP_i, xiP_i), sigT

    def plastic_update(_):
        tol = 1e-8
        max_iter = 10

        def cond_fun(carry):
            epP, xiP, norm_r, iter_count = carry
            return (norm_r > tol) & (iter_count < max_iter)

        def loop_body(carry):
            epP, xiP, _, iter_count = carry
            theta = jnp.stack([epP, xiP])
            rp = residual_vec(theta, ep_new, epP_i, xiP_i, prop).reshape(2, 1)
            Kp = jax.jacfwd(residual_vec, argnums=0)(theta, ep_new, epP_i, xiP_i, prop)
            d_theta = jnp.linalg.solve(-Kp, rp).flatten()
            norm_r = jnp.linalg.norm(rp)
            return (epP + d_theta[0], xiP + d_theta[1], norm_r, iter_count + 1)

        init_val = (epP_i, xiP_i, 1.0, 0)
        final_vals = jax.lax.while_loop(cond_fun, loop_body, init_val)
        epP_final, xiP_final, _, num_iters = final_vals

        xiP_final = jnp.maximum(xiP_final, xiP_i)
        str_eff_final = E * (ep_new - epP_final)
        return (epP_final, xiP_final), str_eff_final

    return jax.lax.cond(phi < 0, elastic_update, plastic_update, operand=None)

@jax.jit
def solve_full_path(strain_history):
    initial_state = (0.0, 0.0)
    def scan_body(carry, strain):
        new_state, stress = material_step(carry, strain)
        return new_state, stress
    _, stress_history = jax.lax.scan(scan_body, initial_state, strain_history)
    return stress_history

# ==========================================
# 2. DATA GENERATORS
# ==========================================
def generate_gp_paths(num_samples, num_steps, max_strain=1.0):
    print(f"Generating {num_samples} Gaussian Process paths...")
    t = np.linspace(0, 1, num_steps).reshape(-1, 1)
    strain_data = np.zeros((num_samples, num_steps))
    num_batches = 10
    batch_size = num_samples // num_batches

    for i in range(num_batches):
        length_scale = np.random.uniform(0.005, 0.05)
        dists = cdist(t, t, metric='sqeuclidean')
        K = np.exp(-dists / (2 * length_scale**2)) + np.eye(num_steps) * 1e-6
        mean = np.zeros(num_steps)
        batch_paths = np.random.multivariate_normal(mean, K, batch_size)
        batch_paths -= batch_paths[:, 0:1]
        max_vals = np.max(np.abs(batch_paths), axis=1, keepdims=True)
        max_vals[max_vals < 1e-6] = 1.0
        target_scales = np.random.uniform(0.1, max_strain, size=(batch_size, 1))
        batch_paths = (batch_paths / max_vals) * target_scales

        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if i == num_batches - 1:
            strain_data[start_idx:] = batch_paths[:num_samples-start_idx]
        else:
            strain_data[start_idx:end_idx] = batch_paths

    return jnp.array(strain_data[..., None])

def generate_random_zigzag_paths(num_samples, num_steps, max_strain=1.0):
    key_points = np.linspace(0, num_steps-1, 7, dtype=int)
    strain_data = np.zeros((num_samples, num_steps))
    for i in range(num_samples):
        checkpoints = np.random.uniform(-max_strain, max_strain, size=5)
        values = np.concatenate(([0], checkpoints, [0]))
        strain_data[i, :] = np.interp(np.arange(num_steps), key_points, values)
    print(f"Generated {num_samples} random Zig-Zag paths for Testing.")
    return jnp.array(strain_data[..., None])

def generate_sinusoidal_paths(num_samples, num_steps, max_strain=1.0, T_total=1.0):
    """Generate random sinusoidal strain paths with varying frequencies and amplitudes."""
    print(f"Generating {num_samples} sinusoidal paths...")
    t = np.linspace(0, T_total, num_steps)
    strain_data = np.zeros((num_samples, num_steps))
    for i in range(num_samples):
        freq = np.random.uniform(1, 5)
        amplitude = np.random.uniform(0.1, max_strain)
        strain_data[i, :] = amplitude * np.abs(np.sin(2 * np.pi * freq * t / T_total))
        strain_data[i, 0] = 0.0
    return jnp.array(strain_data[..., None])

# ==========================================
# 3. MODEL COMPONENTS
# ==========================================

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
        w_trunc    = w_complex[:, :, :actual_modes]
        out_ft = jnp.einsum("bmi,iom->bmo", x_ft_trunc, w_trunc)
        out_ft_padded = jnp.zeros((B, total_freqs, self.out_channels), dtype=jnp.complex64)
        out_ft_padded = out_ft_padded.at[:, :actual_modes, :].set(out_ft)
        x = jnp.fft.irfft(out_ft_padded, n=T, axis=1)
        return x


# ==========================================
# 3b. CAUSAL ATTENTION
# ==========================================
class CausalMultiHeadSelfAttention(nnx.Module):
    """
    Multi-head self-attention with a strict causal mask.
    Output at position t depends only on inputs at positions <= t.
    """
    def __init__(self, dim, num_heads=4, dropout_rate=0.1, rngs: nnx.Rngs = None):
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


class FNOAttnBlock(nnx.Module):
    """
    One block: causal attention (residual, pre-norm) + FNO sub-block
    (spectral conv + pointwise SIREN, sin activation, dropout).
    """
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
        # 1. Causal attention (residual, pre-norm).
        x = x + self.attn(self.attn_norm(x), training=training)
        # 2. FNO sub-block.
        x_spec = self.spectral(x)
        x_loc = self.w(x)
        x = jnp.sin(x_spec + x_loc)
        x = self.dropout(x, deterministic=not training)
        return x


# ==========================================
# 3c. FNO+SIREN+ATTENTION (MAIN MODEL)
# ==========================================
class FNO1D_SIREN_Attn(nnx.Module):
    """Attention-augmented FNO+SIREN."""
    def __init__(self, modes, width, depth, rngs: nnx.Rngs, padding_frac=0.1,
                 omega0=30.0,num_heads=4, dropout_rate=0.1):
        self.width = width
        self.modes = modes
        self.padding_frac = padding_frac
        lifting_in_dim = 1

        self.fc0 = SirenLinear(lifting_in_dim, width, is_first=True, omega0=omega0, rngs=rngs)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(
                FNOAttnBlock(
                    width=width, modes=modes, omega0=omega0,
                    num_heads=num_heads, dropout_rate=dropout_rate, rngs=rngs,
                )
            )

        self.fc1 = SirenLinear(width, 128, is_first=False, omega0=omega0, rngs=rngs)
        self.fc2 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x, training=False):
        B, T, C = x.shape
        x = self.fc0(x)
        x = jnp.sin(x)
        n_pad = int(T * self.padding_frac)
        if n_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)))
        for block in self.blocks:
            x = block(x, training=training)
        if n_pad > 0:
            x = x[:, :T, :]
        x = self.fc1(x)
        x = jnp.sin(x)
        x = self.fc2(x)
        x = x - x[:, 0:1, :]
        return x


# ==============================================================================
# MODEL SAVING AND LOADING
# ==============================================================================
def save_model(state, model_config, filepath):
    """Save model parameters and configuration."""
    print(f"\n   Saving model to {filepath}...")
    params_dict = state.params.to_pure_dict()
    serialized_params = serialization.to_bytes(params_dict)
    checkpoint = {
        'params': serialized_params,
        'model_config': model_config,
        'optimizer_state': state.opt_state,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"   ✓ Model saved successfully!")

def load_model(filepath, model_template):
    """Load model parameters and configuration."""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    template_state = nnx.state(model_template, nnx.Param)
    template_dict = template_state.to_pure_dict()
    params_dict = serialization.from_bytes(template_dict, checkpoint['params'])
    template_state.replace_by_pure_dict(params_dict)
    nnx.update(model_template, template_state)
    opt_state = checkpoint.get('optimizer_state', None)
    return model_template, checkpoint['model_config'], opt_state



def build_model(modes, width, depth, omega0, num_heads, dropout_rate, seed=0):
    """Instantiate a fresh FNO1D_SIREN_Attn model."""
    return FNO1D_SIREN_Attn(
        modes=modes, width=width, depth=depth,
        rngs=nnx.Rngs(seed), omega0=omega0,
        num_heads=num_heads, dropout_rate=dropout_rate,
    )


def train_model(model, opt_state, optimizer,
                x_train, y_train, x_val, y_val, x_test, y_test,
                batch_size, epochs, patience,
                trial=None, log_every=50,
                record_history=True, verbose=True):
    """
    Generic training loop used by both Optuna trials and the final retrain.
    """

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
    n_val   = x_val.shape[0]
    n_test  = x_test.shape[0]
    n_train_b = n_train // batch_size
    n_val_b   = math.ceil(n_val   / batch_size)
    n_test_b  = math.ceil(n_test  / batch_size)

    best_val        = float('inf')
    best_params     = None
    best_epoch      = 0
    patience_counter = 0
    history = {'train': [], 'val': [], 'test': []}

    x_train_j = jnp.array(x_train)
    y_train_j = jnp.array(y_train)

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        x_s  = x_train_j[perm]
        y_s  = y_train_j[perm]

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
            best_val         = val_f
            best_params      = nnx.state(model)
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % log_every == 0:
            extra = f" | test {history['test'][-1]:.5f}" if record_history else ""
            print(f"  Epoch {epoch:5d} | train {float(train_loss):.5f} "
                  f"| val {val_f:.5f}{extra}")

        # Optuna pruning hook
        if trial is not None:
            trial.report(val_f, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_counter >= patience:
            if verbose:
                print(f"  !! Early stop at epoch {epoch}. Best val {best_val:.6f}")
            break

    return best_val, best_params, history, best_epoch, opt_state


# ==========================================
# COMPREHENSIVE STATISTICAL TESTING
# ==========================================
def comprehensive_statistical_testing(model, num_steps=100, results_dir="results"):
    print("\n" + "="*60)
    print("COMPREHENSIVE STATISTICAL TESTING")
    print("="*60)

    T_TOTAL = 1.0
    time_axis = np.linspace(0, T_TOTAL, num_steps)
    batch_solver = jax.vmap(solve_full_path)

    errors_zigzag = []
    errors_sinusoidal = []
    errors_gp = []

    print("\n[1/3] Testing 100 Zig-Zag Paths...")
    zigzag_inputs = generate_random_zigzag_paths(100, num_steps, max_strain=1.0)
    chunk_size = 20
    zigzag_truth = []
    for i in range(0, 100, chunk_size):
        chunk = zigzag_inputs[i:i+chunk_size, :, 0]
        zigzag_truth.append(batch_solver(chunk))
    zigzag_truth = jnp.concatenate(zigzag_truth, axis=0)[..., None]
    zigzag_pred = model(zigzag_inputs, training=False)
    for i in range(100):
        error = jnp.sqrt(jnp.mean((zigzag_pred[i] - zigzag_truth[i])**2)) / jnp.sqrt(jnp.mean(zigzag_truth[i]**2))
        errors_zigzag.append(float(error))
    print(f"  ✓ Zig-Zag: Mean Rel L2 = {np.mean(errors_zigzag):.6f}, Std = {np.std(errors_zigzag):.6f}")

    print("\n[2/3] Testing 100 Sinusoidal Paths...")
    sinusoidal_inputs = generate_sinusoidal_paths(100, num_steps, max_strain=1.0, T_total=T_TOTAL)
    sinusoidal_truth = []
    for i in range(0, 100, chunk_size):
        chunk = sinusoidal_inputs[i:i+chunk_size, :, 0]
        sinusoidal_truth.append(batch_solver(chunk))
    sinusoidal_truth = jnp.concatenate(sinusoidal_truth, axis=0)[..., None]
    sinusoidal_pred = model(sinusoidal_inputs, training=False)
    for i in range(100):
        error = jnp.sqrt(jnp.mean((sinusoidal_pred[i] - sinusoidal_truth[i])**2)) / jnp.sqrt(jnp.mean(sinusoidal_truth[i]**2))
        errors_sinusoidal.append(float(error))
    print(f"  ✓ Sinusoidal: Mean Rel L2 = {np.mean(errors_sinusoidal):.6f}, Std = {np.std(errors_sinusoidal):.6f}")

    print("\n[3/3] Testing 100 Gaussian Process Paths...")
    gp_inputs = generate_gp_paths(100, num_steps, max_strain=1.0)
    gp_truth = []
    for i in range(0, 100, chunk_size):
        chunk = gp_inputs[i:i+chunk_size, :, 0]
        gp_truth.append(batch_solver(chunk))
    gp_truth = jnp.concatenate(gp_truth, axis=0)[..., None]
    gp_pred = model(gp_inputs, training=False)
    for i in range(100):
        error = jnp.sqrt(jnp.mean((gp_pred[i] - gp_truth[i])**2)) / jnp.sqrt(jnp.mean(gp_truth[i]**2))
        errors_gp.append(float(error))
    print(f"  ✓ Gaussian Process: Mean Rel L2 = {np.mean(errors_gp):.6f}, Std = {np.std(errors_gp):.6f}")

    print("\n[4/4] Creating statistical plots...")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, max(max(errors_zigzag), max(errors_sinusoidal), max(errors_gp)), 30)
    plt.hist(errors_zigzag, bins=bins, alpha=0.6, label='Zig-Zag', color='blue', edgecolor='black')
    plt.hist(errors_sinusoidal, bins=bins, alpha=0.6, label='Sinusoidal', color='red', edgecolor='black')
    plt.hist(errors_gp, bins=bins, alpha=0.6, label='Gaussian Process', color='green', edgecolor='black')
    plt.xlabel('Relative L2 Error', fontsize=12); plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution Comparison (100 samples each)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    data_to_plot = [errors_zigzag, errors_sinusoidal, errors_gp]
    bp = plt.boxplot(data_to_plot, labels=['Zig-Zag', 'Sinusoidal', 'Gaussian Process'],
                     patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    plt.ylabel('Relative L2 Error', fontsize=12)
    plt.title('Error Distribution Box Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comprehensive_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    test_names  = ['Zig-Zag', 'Sinusoidal', 'Gaussian Process']
    error_lists = [errors_zigzag, errors_sinusoidal, errors_gp]
    colors_list = ['blue', 'red', 'green']
    for idx, (ax, name, errors, color) in enumerate(zip(axes, test_names, error_lists, colors_list)):
        ax.plot(range(1, 101), sorted(errors), marker='o', color=color, alpha=0.6, markersize=3)
        ax.axhline(y=np.mean(errors), color='black', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(errors):.5f}')
        ax.fill_between(range(1, 101),
                        np.mean(errors) - np.std(errors),
                        np.mean(errors) + np.std(errors),
                        alpha=0.2, color=color, label=f'±1σ = {np.std(errors):.5f}')
        ax.set_xlabel('Sample (sorted by error)', fontsize=11)
        ax.set_ylabel('Relative L2 Error', fontsize=11)
        ax.set_title(f'{name}\n(n=100)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/individual_error_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(f"{results_dir}/statistical_summary.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE STATISTICAL TESTING RESULTS\n")
        f.write("FNO+SIREN+Attention — isotropic plasticity model\n")
        f.write("="*70 + "\n\n")
        for name, errors in zip(test_names, error_lists):
            f.write(f"{name.upper()} (n=100)\n" + "-"*50 + "\n")
            f.write(f"  Mean Relative L2 Error:     {np.mean(errors):.8f}\n")
            f.write(f"  Std Dev:                    {np.std(errors):.8f}\n")
            f.write(f"  Min Error:                  {np.min(errors):.8f}\n")
            f.write(f"  Max Error:                  {np.max(errors):.8f}\n")
            f.write(f"  Median Error:               {np.median(errors):.8f}\n")
            f.write(f"  25th Percentile:            {np.percentile(errors, 25):.8f}\n")
            f.write(f"  75th Percentile:            {np.percentile(errors, 75):.8f}\n")
            f.write(f"  95th Percentile:            {np.percentile(errors, 95):.8f}\n\n")

    print("\n[5/5] Creating example prediction plots...")
    for category_name, inputs, truths, preds in [
        ('zigzag',      zigzag_inputs,      zigzag_truth,      zigzag_pred),
        ('sinusoidal',  sinusoidal_inputs,  sinusoidal_truth,  sinusoidal_pred),
        ('gp',          gp_inputs,          gp_truth,          gp_pred),
    ]:
        indices = np.random.choice(100, 3, replace=False)
        for i, idx in enumerate(indices):
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(time_axis, inputs[idx].flatten(), 'b-', linewidth=2)
            plt.xlabel('Time (s)'); plt.ylabel('Strain')
            plt.title(f'{category_name.upper()} Sample {idx}: Input Strain')
            plt.grid(True, alpha=0.3)
            plt.subplot(2, 2, 2)
            plt.plot(time_axis, truths[idx].flatten(), 'g-', linewidth=2, label='Ground Truth')
            plt.plot(time_axis, preds[idx].flatten(), 'r--', linewidth=2, label='FNO Prediction')
            plt.xlabel('Time (s)'); plt.ylabel('Stress (MPa)')
            plt.title('Stress Response'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.subplot(2, 2, 3)
            plt.plot(inputs[idx].flatten(), truths[idx].flatten(), 'g-', linewidth=2, label='Truth')
            plt.plot(inputs[idx].flatten(), preds[idx].flatten(), 'r--', linewidth=2, label='Pred')
            plt.xlabel('Strain'); plt.ylabel('Stress (MPa)')
            plt.title('Hysteresis Loop'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.subplot(2, 2, 4)
            error_trace = np.abs(truths[idx].flatten() - preds[idx].flatten())
            rl2 = np.sqrt(np.mean(error_trace**2)) / np.sqrt(np.mean(truths[idx].flatten()**2))
            plt.plot(time_axis, error_trace, 'orange', linewidth=2)
            plt.axhline(y=np.mean(error_trace), color='red', linestyle='--',
                        label=f'Rel L2 = {rl2:.5f}')
            plt.xlabel('Time (s)'); plt.ylabel('Absolute Error')
            plt.title('Prediction Error'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/comprehensive_{category_name}_sample_{i}.png", dpi=150)
            plt.close()

    print(f"\n✓ Statistical plots saved to {results_dir}/")
    print("\n" + "="*60 + "\nSTATISTICAL TESTING COMPLETE\n" + "="*60)
    return {'zigzag': errors_zigzag, 'sinusoidal': errors_sinusoidal, 'gp': errors_gp}


# ==========================================
# DISCRETIZATION SENSITIVITY TEST
# ==========================================
def generate_canonical_zigzag_paths(num_samples, num_key_points=7,
                                     max_strain=1.0, seed=42):
    np.random.seed(seed)
    key_points = np.linspace(0, 1, num_key_points)
    key_values = []
    for _ in range(num_samples):
        interior_points = np.random.uniform(-max_strain, max_strain, size=num_key_points - 2)
        end_point = np.random.uniform(0, max_strain)
        values = np.concatenate(([0], interior_points, [end_point]))
        key_values.append(values)
    return key_points, np.array(key_values)


def generate_resampled_path(key_points, key_values, num_steps):
    step_positions = np.linspace(0, 1, num_steps)
    return np.interp(step_positions, key_points, key_values)


def test_discretization_sensitivity(model, results_dir,
                                    resolutions=[50, 60, 70, 80, 90, 100, 120, 140, 160, 200],
                                    num_test_samples=50, training_resolution=100, seed=12345):
    print("\n" + "="*70)
    print("DISCRETIZATION SENSITIVITY TESTING")
    print("="*70)

    key_points, key_values = generate_canonical_zigzag_paths(
        num_test_samples, num_key_points=7, max_strain=1.0, seed=seed)

    results = {
        'resolutions': resolutions, 'training_resolution': training_resolution,
        'mean_error': [], 'std_error': [], 'individual_errors': {}
    }

    for res_idx, num_steps in enumerate(resolutions):
        print(f"\n  [{res_idx+1}/{len(resolutions)}] Resolution {num_steps} steps ...")
        test_strains = np.zeros((num_test_samples, num_steps))
        for i in range(num_test_samples):
            test_strains[i] = generate_resampled_path(key_points, key_values[i], num_steps)
        test_strains_jax = jnp.array(test_strains[..., None])
        bs = jax.vmap(solve_full_path)
        test_stresses = bs(test_strains_jax[..., 0])
        predictions = model(test_strains_jax, training=False)[..., 0]

        individual_errors = []
        for i in range(num_test_samples):
            mse_i = np.mean((predictions[i] - test_stresses[i]) ** 2)
            rms_target = np.sqrt(np.mean(test_stresses[i] ** 2))
            rel_error = np.sqrt(mse_i) / rms_target if rms_target > 1e-10 else np.sqrt(mse_i)
            individual_errors.append(float(rel_error))

        
        results['mean_error'].append(np.mean(individual_errors))
        results['std_error'].append(np.std(individual_errors))
        results['individual_errors'][num_steps] = individual_errors
        print(f" Mean: {np.mean(individual_errors)*100:.2f}%")

    results['mean_error'] = np.array(results['mean_error'])
    results['std_error'] = np.array(results['std_error'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1 = axes[0, 0]
    ax1.plot(resolutions, results['mean_error'] * 100, 'bo-', linewidth=2, markersize=8)
    ax1.fill_between(resolutions,
                     (results['mean_error'] - results['std_error']) * 100,
                     (results['mean_error'] + results['std_error']) * 100,
                     alpha=0.3, color='blue', label='±1σ')
    if training_resolution in resolutions:
        ti = resolutions.index(training_resolution)
        ax1.axvline(x=training_resolution, color='red', linestyle='--', linewidth=2,
                    label=f'Training ({training_resolution})')
        ax1.scatter([training_resolution], [results['mean_error'][ti] * 100],
                    color='red', s=150, zorder=5, marker='*')
    ax1.set_xlabel('Temporal Resolution'); ax1.set_ylabel('MSE (%)')
    ax1.set_title('Discretization Sensitivity'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    bp = ax2.boxplot([results['individual_errors'][r] for r in resolutions],
                     labels=[str(r) for r in resolutions], patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], plt.cm.coolwarm(np.linspace(0, 1, len(resolutions)))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_xlabel('Temporal Resolution'); ax2.set_ylabel('Relative L2')
    ax2.set_title('Error Distribution by Resolution'); ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[1, 0]
    if training_resolution in resolutions:
        ti    = resolutions.index(training_resolution)
        base  = results['mean_error'][ti]
        rel_c = (results['mean_error'] - base) / base * 100
        colors_bar = ['green' if rc <= 0 else 'red' for rc in rel_c]
        ax3.bar(range(len(resolutions)), rel_c, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(resolutions)))
        ax3.set_xticklabels([str(r) for r in resolutions], rotation=45)
        ax3.axhline(y=0, color='black', linewidth=1)
        ax3.set_xlabel('Temporal Resolution')
        ax3.set_ylabel('Relative Change in Mean Error (%)')
        ax3.set_title(f'Change vs Training Resolution ({training_resolution})')
        ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    em = np.array([results['individual_errors'][r] for r in resolutions]).T
    im = ax4.imshow(em, aspect='auto', cmap='YlOrRd',
                    extent=[0, len(resolutions), num_test_samples, 0])
    ax4.set_xticks(np.arange(len(resolutions)) + 0.5)
    ax4.set_xticklabels([str(r) for r in resolutions], rotation=45)
    ax4.set_xlabel('Temporal Resolution'); ax4.set_ylabel('Test Sample Index')
    ax4.set_title('Individual Errors Heatmap')
    plt.colorbar(im, ax=ax4, label='Relative L2 Error')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/discretization_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(f"{results_dir}/discretization_sensitivity_summary.txt", 'w') as f:
        f.write("="*70 + "\nFNO DISCRETIZATION SENSITIVITY RESULTS\n" + "="*70 + "\n\n")
        f.write(f"Training Resolution: {training_resolution}\n")
        f.write(f"{'Resolution':<12}{'Mean (%)':<15}{'Std (%)':<10}\n")
        f.write("-"*50 + "\n")
        for i, res in enumerate(resolutions):
            mk = " <-- TRAINING" if res == training_resolution else ""
            f.write(f"{res:<12}{results['mean_error'][i]*100:<15.2f}"
                    f"{results['std_error'][i]*100:<10.2f}{mk}\n")

    print(f"\n✓ Saved to {results_dir}/discretization_sensitivity*.png/txt")
    return results


def save_timing_info(results_dir, training_time, test_time, num_test_samples, history, best_epoch):
    with open(f"{results_dir}/timing_info.txt", 'w') as f:
        f.write("="*70 + "\nMODEL TIMING INFORMATION\n" + "="*70 + "\n\n")
        f.write("TRAINING:\n" + "-"*70 + "\n")
        f.write(f"  Total training time:        {training_time:.2f} s ({training_time/60:.2f} min)\n")
        f.write(f"  Number of epochs:           {len(history['train'])}\n")
        f.write(f"  Best epoch:                 {best_epoch + 1}\n")
        if len(history['train']) > 0:
            f.write(f"  Average time per epoch:     {training_time/len(history['train']):.2f} s\n")
            f.write(f"  Best validation loss:       {min(history['val']):.6f}\n\n")
        f.write("TESTING:\n" + "-"*70 + "\n")
        f.write(f"  Number of test samples:     {num_test_samples}\n")
        f.write(f"  Total testing time:         {test_time:.2f} s\n")
        f.write(f"  Average per sample:         {test_time/num_test_samples*1000:.2f} ms\n\n")
        f.write(f"TOTAL: {training_time + test_time:.2f} s ({(training_time + test_time)/60:.2f} min)\n")
    print(f"   Saved timing info to {results_dir}/timing_info.txt")


# ==========================================
# x*sin(x) TEST CASE
# ==========================================
def test_xsinx_case(model, N_STEPS, T_TOTAL, results_dir):
    print("\n--- EXTRA TEST: x*sin(x) Strain Path ---")
    x = np.linspace(0, 3 * np.pi, N_STEPS)
    raw = x * np.sin(x)
    raw = raw - raw[0]
    if np.max(np.abs(raw)) > 1e-8:
        raw = raw / np.max(np.abs(raw))
    strain_path = raw
    t = np.linspace(0, T_TOTAL, N_STEPS)
    stress_gt   = np.array(solve_full_path(jnp.array(strain_path)))
    stress_pred = np.array(model(jnp.array(strain_path.reshape(1, N_STEPS, 1)),
                                 training=False)[0, :, 0])
    rel_l2 = np.sqrt(np.mean((stress_pred - stress_gt)**2)) / np.sqrt(np.mean(stress_gt**2))
    print(f"  x*sin(x) Relative L2 Error: {rel_l2:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t, strain_path, color='blue', linewidth=2)
    axes[0, 0].set_title("x·sin(x) Input Strain"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(t, stress_gt,   'g-',  linewidth=2, label='Ground Truth')
    axes[0, 1].plot(t, stress_pred, 'r--', linewidth=2, label='FNO Prediction')
    axes[0, 1].set_title("Stress Response"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(strain_path, stress_gt,   'g-',  linewidth=2, label='Truth')
    axes[1, 0].plot(strain_path, stress_pred, 'r--', linewidth=2, label='Pred')
    axes[1, 0].set_title("Hysteresis Loop"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    abs_err = np.abs(stress_gt - stress_pred)
    axes[1, 1].plot(t, abs_err, color='orange', linewidth=2)
    axes[1, 1].axhline(np.mean(abs_err), color='red', linestyle='--',
                       label=f'Rel. L2 = {rel_l2:.5f}')
    axes[1, 1].set_title("Absolute Error"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle("Complex Test: x·sin(x) Strain Path", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/prediction_xsinx.png", dpi=150, bbox_inches='tight')
    plt.close()
    return rel_l2


# ==============================================================================
# OPTUNA GLOBALS  (set once; reused in every trial)
# ==============================================================================

# Results directory
RESULTS_DIR = "Material_operator_1D_elastoplasticity"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_STEPS = 50
N_GP    = 10000
N_TEST  = 2000
T_TOTAL = 1.0

TUNE_EPOCHS   = 2000
TUNE_PATIENCE = 300


FINAL_EPOCHS   = 10000
FINAL_PATIENCE = 2000



N_TRIALS    = 100   # number of Optuna trials
OPTUNA_SEED = 42

# Joint (width, num_heads) options — width is always divisible by num_heads,
# so an invalid combination is never proposed.
WIDTH_HEADS_OPTIONS = [
    (16, 2), (16, 4),
    (32, 2), (32, 4), (32, 8),
    (48, 2), (48, 4), (48, 6),
    (64, 2), (64, 4), (64, 8),
    (96, 2), (96, 4), (96, 8),
]

_DATA = {}


def prepare_data():
    """Generate data once and cache it in _DATA for all Optuna trials."""
    print("--- Generating data (once, reused for all trials) ---")
    np.random.seed(0)
    full_train_x = generate_gp_paths(N_GP, N_STEPS, max_strain=1.0)
    test_x       = generate_random_zigzag_paths(N_TEST, N_STEPS, max_strain=1.0)

    bs    = jax.vmap(solve_full_path)
    chunk = 1000

    y_list = []
    for i in range(0, full_train_x.shape[0], chunk):
        y_list.append(bs(full_train_x[i:i + chunk, :, 0]))
    full_train_y = jnp.concatenate(y_list, axis=0)[..., None]

    ty_list = []
    for i in range(0, test_x.shape[0], chunk):
        ty_list.append(bs(test_x[i:i + chunk, :, 0]))
    test_y = jnp.concatenate(ty_list, axis=0)[..., None]

    idx   = np.arange(full_train_x.shape[0])
    np.random.shuffle(idx)
    split = int(0.9 * full_train_x.shape[0])
    _DATA['x_train'] = full_train_x[idx[:split]]
    _DATA['y_train'] = full_train_y[idx[:split]]
    _DATA['x_val']   = full_train_x[idx[split:]]
    _DATA['y_val']   = full_train_y[idx[split:]]
    _DATA['x_test']  = test_x
    _DATA['y_test']  = test_y
    print(f"  train {_DATA['x_train'].shape} | val {_DATA['x_val'].shape} "
          f"| test {_DATA['x_test'].shape}")


def objective(trial):
    """Optuna objective: train for TUNE budget, return best validation MSE."""
    modes        = trial.suggest_categorical("modes",       [4, 8, 12, 16, 20, 24])
    wh_idx       = trial.suggest_categorical("width_heads_idx",
                                             list(range(len(WIDTH_HEADS_OPTIONS))))
    width, num_heads = WIDTH_HEADS_OPTIONS[wh_idx]
    depth        = trial.suggest_int(        "depth",       4, 6)
    lr           = trial.suggest_float(      "lr",          1e-4, 1e-3,  log=True)
    batch_size   = trial.suggest_categorical("batch_size",  [32, 64, 128, 256])
    weight_decay = trial.suggest_float(      "weight_decay",1e-6, 1e-2,  log=True)
    omega0       = trial.suggest_float(      "omega0",      5.0,  30.0)
    dropout_rate = trial.suggest_float(      "dropout_rate",0.0,  0.2)

    trial.set_user_attr("width",     width)
    trial.set_user_attr("num_heads", num_heads)

    # Clamp modes to the sequence length
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


# ==========================================
# 4. MAIN TRAINING SCRIPT
# ==========================================
def main():
    t_global_start = time_module.time()
    print("FNO + SIREN + Causal Attention — Isotropic Plasticity — Optuna Tuning")
    print("=" * 70)

    prepare_data()

    time_axis = np.linspace(0, T_TOTAL, N_STEPS)

    # Plot a sample of the training and test distributions
    plt.figure(figsize=(12, 5))
    for i in range(10):
        idx = np.random.randint(0, _DATA['x_train'].shape[0])
        plt.plot(time_axis, _DATA['x_train'][idx].flatten(), alpha=0.6)
    plt.title(f"Training Data Distribution (GP Paths) — {_DATA['x_train'].shape[0]} train samples")
    plt.xlabel("Time (s)"); plt.ylabel("Strain"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/data_distribution_training.png"); plt.close()

    plt.figure(figsize=(12, 5))
    for i in range(10):
        idx = np.random.randint(0, _DATA['x_test'].shape[0])
        plt.plot(time_axis, _DATA['x_test'][idx].flatten(), alpha=0.6)
    plt.title(f"Test Data Distribution (Zig-Zag) — {_DATA['x_test'].shape[0]} samples")
    plt.xlabel("Time (s)"); plt.ylabel("Strain"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/data_distribution_test.png"); plt.close()

    # ============================================================
    # OPTUNA HYPERPARAMETER SEARCH
    # ============================================================
    print(f"\n--- Running Optuna ({N_TRIALS} trials, TUNING budget) ---")
    print(f"    TUNE_EPOCHS={TUNE_EPOCHS}, TUNE_PATIENCE={TUNE_PATIENCE}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=200),
        study_name="fno_plasticity_attention",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   gc_after_trial=True)

    print("\n--- Optuna done ---")
    print(f"Best val MSE : {study.best_value:.6f}")
    print(f"Best params  : {study.best_params}")
    print(f"Best user attrs: {study.best_trial.user_attrs}")

    # Save Optuna results
    with open(f"{RESULTS_DIR}/optuna_best_params.json", 'w') as f:
        json.dump({
            'best_value':      study.best_value,
            'best_params':     study.best_params,
            'best_user_attrs': study.best_trial.user_attrs,
            'n_trials':        len(study.trials),
        }, f, indent=2)

    all_trials = [{'number': t.number, 'value': t.value,
                   'params': t.params, 'user_attrs': t.user_attrs,
                   'state': str(t.state)} for t in study.trials]
    with open(f"{RESULTS_DIR}/optuna_all_trials.json", 'w') as f:
        json.dump(all_trials, f, indent=2, default=str)

    # Optuna visualisation plots
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

    # ============================================================
    # FINAL RETRAIN with best hyperparameters at FULL budget
    # ============================================================
    best       = study.best_params
    best_attrs = study.best_trial.user_attrs
    MODES        = min(best['modes'], N_STEPS // 2 + 1)
    WIDTH        = best_attrs['width']
    NUM_HEADS    = best_attrs['num_heads']
    DEPTH        = best['depth']
    LEARNING_RATE = best['lr']
    BATCH_SIZE   = best['batch_size']
    WEIGHT_DECAY = best['weight_decay']
    OMEGA_0      = best['omega0']
    DROPOUT_RATE = best['dropout_rate']

    print(f"\n--- Retraining best config at FULL budget ---")
    print(f"    modes={MODES}, width={WIDTH}, heads={NUM_HEADS}, depth={DEPTH}")
    print(f"    lr={LEARNING_RATE:.2e}, bs={BATCH_SIZE}, wd={WEIGHT_DECAY:.2e}")
    print(f"    omega0={OMEGA_0:.2f}, dropout={DROPOUT_RATE:.3f}")
    print(f"    FINAL_EPOCHS={FINAL_EPOCHS}, FINAL_PATIENCE={FINAL_PATIENCE}")

    model = build_model(MODES, WIDTH, DEPTH, OMEGA_0, NUM_HEADS, DROPOUT_RATE,
                        seed=0)
    optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))

    train_start = time_module.time()
    best_val, best_params_state, history, best_epoch, opt_state = train_model(
        model, opt_state, optimizer,
        _DATA['x_train'], _DATA['y_train'],
        _DATA['x_val'],   _DATA['y_val'],
        _DATA['x_test'],  _DATA['y_test'],
        batch_size=BATCH_SIZE,
        epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE,
        trial=None, log_every=10,
        record_history=True, verbose=True,
    )
    training_time = time_module.time() - train_start
    print(f"\n  Final training: {training_time/60:.2f} min, best val MSE = {best_val:.6f}")

    # Restore best weights
    if best_params_state is not None:
        nnx.update(model, best_params_state)

    # Save model
    model_config = {
        'modes': MODES, 'width': WIDTH, 'depth': DEPTH, 'omega0': OMEGA_0,
        'num_heads': NUM_HEADS, 'dropout_rate': DROPOUT_RATE,
        'n_steps': N_STEPS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE,
        'weight_decay': WEIGHT_DECAY, 'architecture': 'FNO1D_SIREN_Attn',
    }
    history['train_loss'] = history['train']
    history['val_loss']   = history['val']
    history['best_epoch'] = best_epoch

    model_state = type('State', (), {
        'params': nnx.state(model, nnx.Param),
        'opt_state': opt_state,
    })()
    save_model(model_state, model_config, f"{RESULTS_DIR}/best_model.pkl")


    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label="Train", color='blue')
    plt.plot(history['val'],   label="Val",   color='green',  linestyle='--')
    plt.plot(history['test'],  label="Test",  color='orange', linestyle=':')
    plt.yscale('log'); plt.title('Loss Curve (FNO + SIREN + Attention, Optuna-tuned)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png"); plt.close()

    # Random test predictions
    test_indices = np.random.choice(_DATA['x_test'].shape[0], 3, replace=False)
    test_preds   = model(_DATA['x_test'][test_indices])
    for i, idx in enumerate(test_indices):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(time_axis, _DATA['x_test'][idx].flatten(), color='blue', linewidth=2)
        plt.xlabel("Time (s)"); plt.ylabel("Strain")
        plt.title(f"Test Sample {idx}: Strain vs Time"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        plt.plot(time_axis, _DATA['y_test'][idx].flatten(), 'g-', linewidth=2, label='Truth')
        plt.plot(time_axis, test_preds[i].flatten(), 'r--', linewidth=2, label='FNO')
        plt.xlabel("Time (s)"); plt.ylabel("Stress [MPa]")
        plt.title(f"Test Sample {idx}: Stress vs Time"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        plt.plot(_DATA['x_test'][idx].flatten(), _DATA['y_test'][idx].flatten(),
                 'g-', linewidth=2, label='Truth')
        plt.plot(_DATA['x_test'][idx].flatten(), test_preds[i].flatten(),
                 'r--', linewidth=2, label='FNO')
        plt.xlabel("Strain"); plt.ylabel("Stress [MPa]")
        plt.title("Hysteresis Loop"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        err = np.abs(_DATA['y_test'][idx].flatten() - test_preds[i].flatten())
        plt.plot(time_axis, err, color='orange', linewidth=2)
        plt.xlabel("Time (s)"); plt.ylabel("Error [MPa]")
        plt.title("Prediction Error vs Time"); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/prediction_test_{i}.png", dpi=150)
        plt.close()

    # Sinusoidal test case
    t2   = np.linspace(0, T_TOTAL, N_STEPS)
    Ga   = 1 * np.abs(np.sin(3 * np.pi * t2 / T_TOTAL))
    Ga_j = jnp.array(Ga.reshape(1, N_STEPS, 1))
    Ga_gt   = solve_full_path(Ga_j[0, :, 0])
    Ga_pred = model(Ga_j)[0, :, 0]
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1); plt.plot(t2, Ga, 'b-', linewidth=2)
    plt.title("Sinusoidal Test: Strain"); plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.plot(t2, Ga_gt, 'g-', linewidth=2, label='Truth')
    plt.plot(t2, Ga_pred, 'r--', linewidth=2, label='FNO')
    plt.title("Stress"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    plt.plot(Ga, Ga_gt, 'g-', linewidth=2); plt.plot(Ga, Ga_pred, 'r--', linewidth=2)
    plt.title("Hysteresis"); plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    plt.plot(t2, np.abs(Ga_gt - Ga_pred), 'orange', linewidth=2)
    plt.title("Absolute Error"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/prediction_sinusoidal.png", dpi=150); plt.close()

    # x*sin(x)
    test_xsinx_case(model, N_STEPS, T_TOTAL, RESULTS_DIR)

    # 5 random zig-zag plots
    batch_solver = jax.vmap(solve_full_path)
    num_random_tests = 5
    kp = np.linspace(0, N_STEPS-1, 7, dtype=int)
    rand_strains = np.zeros((num_random_tests, N_STEPS))
    for i in range(num_random_tests):
        checkpoints = np.random.uniform(-1.0, 1.0, size=5)
        rand_strains[i, :] = np.interp(np.arange(N_STEPS), kp,
                                        np.concatenate(([0], checkpoints, [0])))
    rand_j = jnp.array(rand_strains[..., None])
    rand_gt   = batch_solver(rand_j[..., 0])
    rand_pred = model(rand_j)[..., 0]
    for i in range(num_random_tests):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1); plt.plot(time_axis, rand_strains[i], 'b-', linewidth=2)
        plt.title(f"Random Zig-Zag {i}: Strain"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        plt.plot(time_axis, rand_gt[i], 'g-', linewidth=2, label='Truth')
        plt.plot(time_axis, rand_pred[i], 'r--', linewidth=2, label='FNO')
        plt.title(f"Stress"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        plt.plot(rand_strains[i], rand_gt[i], 'g-', linewidth=2)
        plt.plot(rand_strains[i], rand_pred[i], 'r--', linewidth=2)
        plt.title("Hysteresis"); plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        plt.plot(time_axis, np.abs(rand_gt[i] - rand_pred[i]), 'orange', linewidth=2)
        plt.title("Error"); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/prediction_random_zigzag_{i}.png", dpi=150); plt.close()

    # GP eval samples
    N_EASY_TEST = 10
    gp_in = generate_gp_paths(N_EASY_TEST, N_STEPS, max_strain=1.0)
    y_true = batch_solver(gp_in[..., 0])[..., None]
    y_pred = model(gp_in, training=False)
    for i in range(N_EASY_TEST):
        loss = jnp.mean((y_pred[i] - y_true[i])**2)
        print(f"GP Sample {i}: MSE = {loss:.6f}")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1); plt.plot(time_axis, gp_in[i].flatten(), 'b-')
        plt.title("Input Strain")
        plt.subplot(1, 3, 2)
        plt.plot(time_axis, y_true[i].flatten(), 'g-', label='Truth')
        plt.plot(time_axis, y_pred[i].flatten(), 'r--', label='Pred')
        plt.legend(); plt.title("Stress Response")
        plt.subplot(1, 3, 3)
        plt.plot(gp_in[i].flatten(), y_true[i].flatten(), 'g-')
        plt.plot(gp_in[i].flatten(), y_pred[i].flatten(), 'r--')
        plt.title("Hysteresis")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/eval_gp_sample_{i}.png"); plt.close()

    test_t0 = time_module.time()

    print("\n--- Comprehensive Statistical Testing ---")
    error_results = comprehensive_statistical_testing(
        model, num_steps=N_STEPS, results_dir=RESULTS_DIR)

    print("\n--- Discretization Sensitivity Study ---")
    test_discretization_sensitivity(
        model=model, results_dir=RESULTS_DIR,
        resolutions=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000],
        num_test_samples=100, training_resolution=N_STEPS, seed=12345,
    )

    test_time = time_module.time() - test_t0
    save_timing_info(RESULTS_DIR, training_time, test_time, 300, history, best_epoch)

    total = time_module.time() - t_global_start
    print("\n" + "="*70)
    print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nTotal elapsed time: {total/60:.1f} min")
    print(f"Results saved in:   {RESULTS_DIR}/")
    print("\nKey output files:")
    print("  optuna_best_params.json          — best hyperparameters")
    print("  optuna_all_trials.json           — all trial results")
    print("  optuna_history.png               — optimisation history")
    print("  optuna_importances.png           — hyperparameter importances")
    print("  optuna_parallel.png              — parallel coordinate plot")
    print("  optuna_slice.png                 — slice plot")
    print("  best_model.pkl                   — trained model checkpoint")
    print("  loss_curve.png                   — final retrain loss")
    print("  comprehensive_error_distribution.png")
    print("  individual_error_distributions.png")
    print("  statistical_summary.txt")
    print("  discretization_sensitivity.png")
    print("  discretization_sensitivity_summary.txt")
    print("  timing_info.txt")


if __name__ == "__main__":
    main()