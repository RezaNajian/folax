"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE
"""
# fol/tools/newton_residual_tracker.py
import os
import numpy as np
import jax.numpy as jnp


class NewtonResidualTracker:
    """
    Tracks Newton residuals for a single FE solve and writes them to CSV.
    Residuals are always taken on the unknown (non-Dirichlet) DOFs.
    """

    def __init__(self, case_dir, sample_tag="baseline"):
        self.path = os.path.join(case_dir, f"newton_residuals_{sample_tag}.csv")
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
):
    """
    Newton–Raphson with:
    - tracking of L2 and RMS residual on unknown DOFs
    - early-stop when 'target_best' is reached (full-load safe by default)
    - optional windowed RMS-mean stoppage (per load step)
    - guard against divergence spikes (growth_tol)

    Returns:
        best_dofs, residuals_rms, total_iters
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

    for fac in range(n_incr):
        load_factor = (fac + 1) / n_incr
        applied_dofs = fe_loss.ApplyDirichletBCOnDofVector(applied_dofs, load_factor)

        prev_res = None
        step_residuals_rms = []  # per-increment residual history (for windowed RMS-mean)

        for it in range(1, maxit + 1):
            total_iters += 1

            # FE residual/Jacobian at current state
            jac, res = fe_loss.ComputeJacobianMatrixAndResidualVector(
                control_vars, applied_dofs
            )

            # Restrict to unknown DOFs
            res_unknown = res[nd]
            _, res_rms = tracker.log(load_step=fac + 1, it=it, res_unknown_jax=res_unknown)

            residuals_rms.append(res_rms)
            step_residuals_rms.append(res_rms)

            # Optional: windowed RMS-mean for smoother convergence decision
            rmsmean = None
            if rmsmean_window and len(step_residuals_rms) >= rmsmean_window:
                rmsmean = _rms_mean(step_residuals_rms[-rmsmean_window:])

            # Non-finite → rollback and stop
            if not np.isfinite(res_rms):
                print(
                    f"[NH-guard] non-finite residual at step {fac+1}, it {it}. "
                    f"Rollback to best and stop."
                )
                if fac == n_incr - 1 and best_dofs_full is not None:
                    return best_dofs_full, residuals_rms, total_iters
                return best_dofs_all, residuals_rms, total_iters

            # Update best-so-far (global)
            if res_rms < best_res_all:
                best_res_all = res_rms
                best_step_all = (fac + 1, it)
                best_dofs_all = jnp.array(applied_dofs)

            # Update best-so-far at FULL LOAD (preferred)
            if fac == n_incr - 1 and res_rms < best_res_full:
                best_res_full = res_rms
                best_step_full = (fac + 1, it)
                best_dofs_full = jnp.array(applied_dofs)

            # Early-stop logic
            can_stop = (not stop_only_at_full_load) or (fac == n_incr - 1)
            if can_stop:
                # (A) windowed RMS-mean criterion (if enabled)
                if rmsmean is not None and rmsmean <= target_rmsmean:
                    print(
                        f"[NH-early] windowed rms-mean reached: rmsmean(w={rmsmean_window})={rmsmean:.3e} "
                        f"<= {target_rmsmean:.3e} at step {fac+1}, it {it}."
                    )
                    return (
                        best_dofs_full if best_dofs_full is not None else jnp.array(applied_dofs)
                    ), residuals_rms, total_iters

                # (B) best RMS residual criterion (full-load safe)
                best_for_stop = (
                    best_res_full
                    if (stop_only_at_full_load and fac == n_incr - 1)
                    else best_res_all
                )
                if best_for_stop <= target_best:
                    if stop_only_at_full_load and best_dofs_full is not None:
                        step_txt = f"{best_step_full[0]}, it {best_step_full[1]}"
                        print(
                            f"[NH-early] target reached: best_rms={best_for_stop:.3e} <= {target_best:.3e} "
                            f"at step {step_txt}."
                        )
                        return best_dofs_full, residuals_rms, total_iters

                    step_txt = f"{best_step_all[0]}, it {best_step_all[1]}"
                    print(
                        f"[NH-early] target reached: best_rms={best_for_stop:.3e} <= {target_best:.3e} "
                        f"at step {step_txt}."
                    )
                    return best_dofs_all, residuals_rms, total_iters

            # Divergence spike → rollback and stop
            if prev_res is not None and res_rms > growth_tol * prev_res:
                print(
                    f"[NH-guard] residual jump from {prev_res:.3e} to {res_rms:.3e} "
                    f"(>{growth_tol}×) at step {fac+1}, it {it}. "
                    f"Rollback to best and stop."
                )
                if fac == n_incr - 1 and best_dofs_full is not None:
                    return best_dofs_full, residuals_rms, total_iters
                return best_dofs_all, residuals_rms, total_iters

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
                    f"[NH-guard] non-finite Newton step at step {fac+1}, it {it}. "
                    f"Rollback to best and stop."
                )
                if fac == n_incr - 1 and best_dofs_full is not None:
                    return best_dofs_full, residuals_rms, total_iters
                return best_dofs_all, residuals_rms, total_iters

            applied_dofs = applied_dofs.at[nd].add(delta[nd])

    # Prefer a best state at FULL LOAD if we reached the last increment.
    if best_dofs_full is not None and np.isfinite(best_res_full):
        print(
            f"[NH-done] finished all steps. "
            f"Best(full-load) rms={best_res_full:.3e} at step {best_step_full[0]}, it {best_step_full[1]}."
        )
        return best_dofs_full, residuals_rms, total_iters

    if np.isfinite(best_res_all):
        print(
            f"[NH-done] finished all steps. "
            f"Best(overall) rms={best_res_all:.3e} at step {best_step_all[0]}, it {best_step_all[1]}."
        )
        return best_dofs_all, residuals_rms, total_iters

    print("[NH-done] no valid best state recorded; returning latest DOFs.")
    return applied_dofs, residuals_rms, total_iters