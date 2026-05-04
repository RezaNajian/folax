"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE
"""
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class NewtonResidualTracker:
    """
    Tracks Newton residuals for a single FE solve and writes them to CSV.
    Residuals are always taken on the unknown (non-Dirichlet) DOFs.
    """

    def __init__(self, case_dir, sample_tag="baseline"):
        self.path = os.path.join(case_dir, f"newton_residuals_{sample_tag}.csv")
        self.case_dir = case_dir
        self.sample_tag = sample_tag
        with open(self.path, "w") as f:
            f.write("load_step,iter,res_l2,res_rms_unknown\n")

    def log(self, load_step, it, res_unknown_jax):
        """
        res_unknown_jax: residual vector restricted to non-Dirichlet DOFs.
        """
        res_unknown = np.asarray(res_unknown_jax)
        l2 = float(np.linalg.norm(res_unknown))
        # Guard against a degenerate case (shouldn't happen in practice)
        rms = float(l2 / np.sqrt(max(1, res_unknown.size)))

        print(
            f"[Newton] step {load_step:03d} it {it:02d} "
            f"||r||_2={l2:.3e}, rms={rms:.3e}"
        )

        with open(self.path, "a") as f:
            f.write(f"{load_step},{it},{l2:.8e},{rms:.8e}\n")

        return l2, rms

    def plot_convergence(self, save_name=None, show=False, figsize=(14, 5)):
        """
        Plot Newton iteration convergence history from the CSV file.
        
        Args:
            save_name: Optional custom filename. If None, uses default naming.
            show: If True, displays the plot interactively.
            figsize: Figure size tuple (width, height).
        """
        # Read the CSV file
        if not os.path.exists(self.path):
            print(f"[Plot] Warning: CSV file {self.path} not found. Skipping plot.")
            return
        
        data = np.genfromtxt(self.path, delimiter=',', skip_header=1)
        
        if data.size == 0:
            print(f"[Plot] Warning: No data in {self.path}. Skipping plot.")
            return
        
        # Handle single row case
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        load_steps = data[:, 0].astype(int)
        iterations = data[:, 1].astype(int)
        res_l2 = data[:, 2]
        res_rms = data[:, 3]
        
        # Create cumulative iteration count for x-axis
        cumulative_iters = np.arange(1, len(iterations) + 1)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        
        # Plot 1: RMS residual vs cumulative iteration
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(cumulative_iters, res_rms, 'o-', linewidth=2, markersize=4, label='RMS residual')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Cumulative Newton Iteration', fontsize=11)
        ax1.set_ylabel('RMS Residual', fontsize=11)
        ax1.set_title('Convergence History', fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Plot 2: Residual per load step (grouped)
        ax2 = fig.add_subplot(gs[0, 1])
        unique_steps = np.unique(load_steps)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_steps)))
        
        for i, step in enumerate(unique_steps):
            mask = load_steps == step
            step_iters = iterations[mask]
            step_rms = res_rms[mask]
            ax2.semilogy(step_iters, step_rms, 'o-', color=colors[i], 
                        linewidth=2, markersize=5, label=f'Step {step}')
        
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Inner Iteration', fontsize=11)
        ax2.set_ylabel('RMS Residual', fontsize=11)
        ax2.set_title('Per Load Step', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8, ncol=2)
        
        # Plot 3: Summary statistics
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Count iterations per load step
        iters_per_step = []
        final_res_per_step = []
        for step in unique_steps:
            mask = load_steps == step
            iters_per_step.append(np.sum(mask))
            final_res_per_step.append(res_rms[mask][-1])
        
        x_pos = np.arange(len(unique_steps))
        bars = ax3.bar(x_pos, iters_per_step, color=colors, alpha=0.7, edgecolor='black')
        
        # Add text on bars showing final residual
        for i, (bar, final_res) in enumerate(zip(bars, final_res_per_step)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{final_res:.1e}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax3.set_xlabel('Load Step', fontsize=11)
        ax3.set_ylabel('Newton Iterations', fontsize=11)
        ax3.set_title('Iterations per Load Step', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{s}' for s in unique_steps])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        total_iters = len(iterations)
        final_rms = res_rms[-1]
        fig.suptitle(f'Newton Solver Convergence: {self.sample_tag} | '
                    f'Total Iterations: {total_iters} | Final RMS: {final_rms:.2e}',
                    fontsize=13, fontweight='bold', y=0.98)
        
        # Save figure
        if save_name is None:
            save_name = f"newton_convergence_{self.sample_tag}.png"
        
        save_path = os.path.join(self.case_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Convergence plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def _rms_mean(values):
    """Windowed RMS-mean: sqrt(mean(values^2))."""
    if values is None or len(values) == 0:
        return np.inf
    v = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(v * v)))


def custom_newton_solve(
    fe_solver,
    control_vars,
    initial_dofs,
    case_dir,
    sample_tag="baseline",
    target_best=1e-6,
    growth_tol=50.0,
    # --- new (backward compatible) knobs ---
    # If >0, compute a per-load-step windowed RMS-mean of the Newton residual
    # and allow early-stop when it drops below target_rmsmean.
    rmsmean_window=5,
    target_rmsmean=None,
    # If True, only allow early-stop at the last load increment (load_factor = 1.0).
    stop_only_at_full_load=True,
    # Optional backtracking line search for robust Newton updates.
    use_line_search=True,
    ls_max_backtracks=6,
    ls_shrink=0.5,
    ls_accept_ratio=0.98,
    guard_return_mode="best",
    plateau_max_consecutive=5,
    plateau_rel_improve_tol=1e-4,
    return_run_info=False,
    # --- adaptive load stepping ---
    # If True, step size grows after easy Newton steps and shrinks after hard ones,
    # minimising the number of load increments needed from a cold (zero) start.
    adaptive_load_incr=False,
    easy_step_iters=2,       # iters <= this  →  grow step
    step_grow_factor=2.0,    # multiply step size after an easy step
    step_shrink_factor=0.5,  # multiply step size after a hard step
):
    """
    Newton–Raphson with:
    - tracking of L2 and RMS residual on unknown DOFs
    - early-stop when 'target_best' is reached (full-load safe by default)
    - optional windowed RMS-mean stoppage (per load step)
    - guard against divergence spikes (growth_tol)
    - optional backtracking line-search on Newton step
    - configurable guard return mode: "best" or "latest"

    Returns:
        best_dofs, residuals_rms, total_iters
        If return_run_info=True, also returns run_info dict.
    """
    fe_loss = fe_solver.fe_loss_function
    nd = fe_loss.non_dirichlet_indices

    tracker = NewtonResidualTracker(case_dir, sample_tag)

    # Normalize thresholds
    if target_rmsmean is None:
        target_rmsmean = float(target_best)
    rmsmean_window = int(rmsmean_window) if rmsmean_window is not None else 0

    residuals_rms = []  # full history across all increments (for plotting)
    applied_dofs = initial_dofs

    n_incr = fe_solver.nonlinear_solver_settings["load_incr"]
    maxit = fe_solver.nonlinear_solver_settings["maxiter"]
    atol = fe_solver.nonlinear_solver_settings["abs_tol"]
    rtol = fe_solver.nonlinear_solver_settings["rel_tol"]

    # Best over *all* steps (fallback)
    best_res_all = np.inf
    best_step_all = (0, 0)
    best_dofs_all = jnp.array(applied_dofs)

    # Best at full load factor (preferred return when available)
    best_res_full = np.inf
    best_step_full = (0, 0)
    best_dofs_full = None

    total_iters = 0

    run_info = {
        "fe_converged": False,
        "reached_full_load": False,
        "failed_load_step": "",
        "reached_load_factor": 0.0,
        "guard_return_mode": str(guard_return_mode),
        "line_search_used": bool(use_line_search),
        "failure_mode": "",
    }

    def _pack_return(dofs):
        if return_run_info:
            return dofs, residuals_rms, total_iters, dict(run_info)
        return dofs, residuals_rms, total_iters

    def _mark_failure(step_num, actual_load_factor):
        run_info["fe_converged"] = False
        run_info["reached_full_load"] = False
        run_info["failed_load_step"] = int(step_num)
        run_info["reached_load_factor"] = float(actual_load_factor)

    def _guard_return_current_or_best(step_num, actual_load_factor, at_full_load, failure_mode):
        _mark_failure(step_num, actual_load_factor)
        run_info["failure_mode"] = str(failure_mode)
        
        # Plot convergence history even on failure
        tracker.plot_convergence()
        
        if guard_return_mode == "latest":
            return _pack_return(jnp.array(applied_dofs))
        if at_full_load and best_dofs_full is not None:
            return _pack_return(best_dofs_full)
        return _pack_return(best_dofs_all)

    # --- load schedule -------------------------------------------------------
    # Fixed:    uniform steps 1/n_incr, 2/n_incr, ..., 1.0  (adaptive_load_incr=False)
    # Adaptive: step size grows/shrinks based on inner iteration count          (adaptive_load_incr=True)
    #   Floor = 1/(4*n_incr) so we never subdivide to infinity.
    _step_size   = 1.0 / n_incr          # initial step size
    _step_floor  = 1.0 / (4 * n_incr)   # minimum step size
    _load_factor = 0.0                   # load level reached so far
    _step_num    = 0                     # step counter (for logging)

    while _load_factor < 1.0 - 1e-12:
        _load_factor  = min(_load_factor + _step_size, 1.0)
        _step_num    += 1
        load_factor   = _load_factor
        at_full_load  = (load_factor >= 1.0 - 1e-12)

        applied_dofs = fe_loss.ApplyDirichletBCOnDofVector(applied_dofs, load_factor)

        prev_res = None
        step_residuals_rms = []  # per-increment residual history (for windowed RMS-mean)
        plateau_streak = 0
        _step_inner_iters = 0   # Newton iters used in this load step
        _hit_maxit = False      # flag: inner loop exhausted maxiter

        for it in range(1, maxit + 1):
            total_iters += 1
            _step_inner_iters += 1

            # FE residual/Jacobian at current state
            jac, res = fe_loss.ComputeJacobianMatrixAndResidualVector(
                control_vars, applied_dofs
            )

            # Restrict to unknown DOFs
            res_unknown = res[nd]
            _, res_rms = tracker.log(load_step=_step_num, it=it, res_unknown_jax=res_unknown)

            residuals_rms.append(res_rms)
            step_residuals_rms.append(res_rms)

            # Optional: windowed RMS-mean for smoother convergence decision
            rmsmean = None
            if rmsmean_window and len(step_residuals_rms) >= rmsmean_window:
                rmsmean = _rms_mean(step_residuals_rms[-rmsmean_window:])

            # Non-finite → rollback and stop
            if not np.isfinite(res_rms):
                print(
                    f"[NH-guard] non-finite residual at step {_step_num}, it {it}. "
                    f"Rollback to best and stop."
                )
                return _guard_return_current_or_best(_step_num, load_factor, at_full_load, "nonfinite_residual")

            # Update best-so-far (global)
            if res_rms < best_res_all:
                best_res_all = res_rms
                best_step_all = (_step_num, it)
                best_dofs_all = jnp.array(applied_dofs)

            # Update best-so-far at FULL LOAD (preferred)
            if at_full_load and res_rms < best_res_full:
                best_res_full = res_rms
                best_step_full = (_step_num, it)
                best_dofs_full = jnp.array(applied_dofs)

            # Early-stop logic
            can_stop = (not stop_only_at_full_load) or at_full_load
            if can_stop:
                # (A) windowed RMS-mean criterion (if enabled)
                if rmsmean is not None and rmsmean <= target_rmsmean:
                    print(
                        f"[NH-early] windowed rms-mean reached: rmsmean(w={rmsmean_window})={rmsmean:.3e} "
                        f"<= {target_rmsmean:.3e} at step {_step_num}, it {it}."
                    )
                    run_info["fe_converged"] = at_full_load
                    run_info["reached_full_load"] = at_full_load
                    run_info["failed_load_step"] = ""
                    run_info["reached_load_factor"] = float(load_factor)
                    run_info["failure_mode"] = "early_stop_rmsmean"
                    
                    # Plot convergence history
                    tracker.plot_convergence()
                    
                    return _pack_return(
                        best_dofs_full if best_dofs_full is not None else jnp.array(applied_dofs)
                    )

                # (B) best RMS residual criterion (full-load safe)
                best_for_stop = (
                    best_res_full
                    if (stop_only_at_full_load and at_full_load)
                    else best_res_all
                )
                if best_for_stop <= target_best:
                    if stop_only_at_full_load and best_dofs_full is not None:
                        step_txt = f"{best_step_full[0]}, it {best_step_full[1]}"
                        print(
                            f"[NH-early] target reached: best_rms={best_for_stop:.3e} <= {target_best:.3e} "
                            f"at step {step_txt}."
                        )
                        run_info["fe_converged"] = True
                        run_info["reached_full_load"] = True
                        run_info["failed_load_step"] = ""
                        run_info["reached_load_factor"] = 1.0
                        run_info["failure_mode"] = "full_load_converged"
                        
                        # Plot convergence history
                        tracker.plot_convergence()
                        
                        return _pack_return(best_dofs_full)

                    step_txt = f"{best_step_all[0]}, it {best_step_all[1]}"
                    print(
                        f"[NH-early] target reached: best_rms={best_for_stop:.3e} <= {target_best:.3e} "
                        f"at step {step_txt}."
                    )
                    run_info["fe_converged"] = at_full_load
                    run_info["reached_full_load"] = at_full_load
                    run_info["failed_load_step"] = "" if at_full_load else int(_step_num)
                    run_info["reached_load_factor"] = float(load_factor)
                    run_info["failure_mode"] = (
                        "full_load_converged" if at_full_load else "early_stop_target"
                    )
                    
                    # Plot convergence history
                    tracker.plot_convergence()
                    
                    return _pack_return(best_dofs_all)

            # Divergence spike → rollback and stop
            if prev_res is not None and res_rms > growth_tol * prev_res:
                print(
                    f"[NH-guard] residual jump from {prev_res:.3e} to {res_rms:.3e} "
                    f"(>{growth_tol}×) at step {_step_num}, it {it}. "
                    f"Rollback to best and stop."
                )
                return _guard_return_current_or_best(_step_num, load_factor, at_full_load, "growth_guard")

            # Convergence / stagnation for this substep
            if res_rms < atol:
                break
            if prev_res is not None and res_rms >= (1.0 - rtol) * prev_res:
                # Relative improvement smaller than rtol → treat as stagnation
                break

            prev_res = res_rms

            # Newton update
            delta = fe_solver.LinearSolve(jac, res, applied_dofs)
            if not jnp.all(jnp.isfinite(delta)):
                print(
                    f"[NH-guard] non-finite Newton step at step {_step_num}, it {it}. "
                    f"Rollback to best and stop."
                )
                return _guard_return_current_or_best(_step_num, load_factor, at_full_load, "nonfinite_delta")

            if use_line_search:
                accepted = False
                alpha = 1.0
                max_backtracks = int(max(0, ls_max_backtracks))
                shrink = float(ls_shrink)
                accept_ratio = float(ls_accept_ratio)
                tried_backtracks = 0
                last_trial_rms = np.inf

                for bt in range(max_backtracks + 1):
                    tried_backtracks = bt
                    trial_dofs = applied_dofs.at[nd].add(alpha * delta[nd])
                    _, trial_res = fe_loss.ComputeJacobianMatrixAndResidualVector(
                        control_vars, trial_dofs
                    )
                    trial_unknown = np.asarray(trial_res[nd])
                    trial_l2 = float(np.linalg.norm(trial_unknown))
                    trial_rms = float(trial_l2 / np.sqrt(max(1, trial_unknown.size)))
                    last_trial_rms = trial_rms

                    if np.isfinite(trial_rms) and trial_rms <= accept_ratio * res_rms:
                        applied_dofs = trial_dofs
                        accepted = True
                        improved_enough = trial_rms < (1.0 - float(plateau_rel_improve_tol)) * res_rms
                        if improved_enough:
                            plateau_streak = 0
                        else:
                            plateau_streak += 1
                            print(
                                f"[NH-plateau] near-flat accepted step at step {_step_num}, it {it}: "
                                f"trial_rms={trial_rms:.3e}, prev_rms={res_rms:.3e}, streak={plateau_streak}."
                            )
                            if plateau_streak >= int(max(1, plateau_max_consecutive)):
                                print(
                                    f"[NH-guard] plateau stagnation at step {_step_num}, it {it}. "
                                    f"Reached {plateau_streak} consecutive near-flat accepted step(s)."
                                )
                                return _guard_return_current_or_best(_step_num, load_factor, at_full_load, "plateau_stagnation")

                        if bt > 0:
                            print(
                                f"[NH-ls] accepted damped step at step {_step_num}, it {it}: "
                                f"alpha={alpha:.3e}, trial_rms={trial_rms:.3e}, prev_rms={res_rms:.3e}."
                            )
                        break

                    if bt < max_backtracks:
                        alpha *= shrink

                if not accepted:
                    print(
                        f"[NH-guard] line-search failed at step {_step_num}, it {it}. "
                        f"No acceptable alpha in {tried_backtracks + 1} trial(s); "
                        f"last_trial_rms={last_trial_rms:.3e}, prev_rms={res_rms:.3e}. "
                        f"Rollback to best and stop."
                    )
                    return _guard_return_current_or_best(_step_num, load_factor, at_full_load, "line_search_failed")
            else:
                applied_dofs = applied_dofs.at[nd].add(delta[nd])

        else:
            _hit_maxit = True

        if _hit_maxit:
            return _guard_return_current_or_best(_step_num, load_factor, at_full_load, "maxiter_reached")

        # --- adaptive step-size update (after inner Newton loop) -------------
        if adaptive_load_incr and not at_full_load:
            if _step_inner_iters <= easy_step_iters:
                _step_size *= step_grow_factor
                print(
                    f"[NH-adapt] step {_step_num} easy ({_step_inner_iters} iters) "
                    f"→ growing step size to {_step_size:.4f}"
                )
            else:
                _step_size *= step_shrink_factor
                print(
                    f"[NH-adapt] step {_step_num} hard ({_step_inner_iters} iters) "
                    f"→ shrinking step size to {_step_size:.4f}"
                )
            # clamp: never smaller than floor, never larger than remaining load
            _step_size = max(_step_size, _step_floor)
            _step_size = min(_step_size, 1.0 - _load_factor)

    # Prefer a best state at FULL LOAD if we reached the last increment.
    if best_dofs_full is not None and np.isfinite(best_res_full):
        print(
            f"[NH-done] finished all steps. "
            f"Best(full-load) rms={best_res_full:.3e} at step {best_step_full[0]}, it {best_step_full[1]}."
        )
        run_info["fe_converged"] = True
        run_info["reached_full_load"] = True
        run_info["failed_load_step"] = ""
        run_info["reached_load_factor"] = 1.0
        run_info["failure_mode"] = "full_load_converged"
        
        # Plot convergence history
        tracker.plot_convergence()
        
        return _pack_return(best_dofs_full)

    if np.isfinite(best_res_all):
        print(
            f"[NH-done] finished all steps. "
            f"Best(overall) rms={best_res_all:.3e} at step {best_step_all[0]}, it {best_step_all[1]}."
        )
        run_info["fe_converged"] = False
        run_info["reached_full_load"] = False
        run_info["failed_load_step"] = _step_num
        run_info["reached_load_factor"] = float(_load_factor)
        run_info["failure_mode"] = "maxiter_reached"
        
        # Plot convergence history
        tracker.plot_convergence()
        
        return _pack_return(best_dofs_all)

    print("[NH-done] no valid best state recorded; returning latest DOFs.")
    run_info["fe_converged"] = False
    run_info["reached_full_load"] = False
    run_info["failed_load_step"] = n_incr
    run_info["reached_load_factor"] = 0.0
    run_info["failure_mode"] = "no_valid_state"
    
    # Plot convergence history
    tracker.plot_convergence()
    
    return _pack_return(applied_dofs)
