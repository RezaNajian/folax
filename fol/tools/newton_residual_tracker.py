"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE
"""
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
    # Optional backtracking line search for robust Newton updates.
    use_line_search=True,
    ls_max_backtracks=6,
    ls_shrink=0.5,
    ls_accept_ratio=0.98,
    guard_return_mode="best",
    plateau_max_consecutive=5,
    plateau_rel_improve_tol=1e-4,
    return_run_info=False,
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

    def _mark_failure(fac_idx):
        run_info["fe_converged"] = False
        run_info["reached_full_load"] = False
        run_info["failed_load_step"] = int(fac_idx + 1)
        run_info["reached_load_factor"] = float((fac_idx + 1) / n_incr)

    def _guard_return_current_or_best(fac_idx, failure_mode):
        _mark_failure(fac_idx)
        run_info["failure_mode"] = str(failure_mode)
        if guard_return_mode == "latest":
            return _pack_return(jnp.array(applied_dofs))
        if fac_idx == n_incr - 1 and best_dofs_full is not None:
            return _pack_return(best_dofs_full)
        return _pack_return(best_dofs_all)

    for fac in range(n_incr):
        load_factor = (fac + 1) / n_incr
        applied_dofs = fe_loss.ApplyDirichletBCOnDofVector(applied_dofs, load_factor)

        prev_res = None
        step_residuals_rms = []  # per-increment residual history (for windowed RMS-mean)
        plateau_streak = 0

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
                return _guard_return_current_or_best(fac, "nonfinite_residual")

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
                    run_info["fe_converged"] = bool(fac == n_incr - 1)
                    run_info["reached_full_load"] = bool(fac == n_incr - 1)
                    run_info["failed_load_step"] = ""
                    run_info["reached_load_factor"] = float((fac + 1) / n_incr)
                    run_info["failure_mode"] = "early_stop_rmsmean"
                    return _pack_return(
                        best_dofs_full if best_dofs_full is not None else jnp.array(applied_dofs)
                    )

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
                        run_info["fe_converged"] = True
                        run_info["reached_full_load"] = True
                        run_info["failed_load_step"] = ""
                        run_info["reached_load_factor"] = 1.0
                        run_info["failure_mode"] = "full_load_converged"
                        return _pack_return(best_dofs_full)

                    step_txt = f"{best_step_all[0]}, it {best_step_all[1]}"
                    print(
                        f"[NH-early] target reached: best_rms={best_for_stop:.3e} <= {target_best:.3e} "
                        f"at step {step_txt}."
                    )
                    run_info["fe_converged"] = bool(fac == n_incr - 1)
                    run_info["reached_full_load"] = bool(fac == n_incr - 1)
                    run_info["failed_load_step"] = "" if fac == n_incr - 1 else int(fac + 1)
                    run_info["reached_load_factor"] = float((fac + 1) / n_incr)
                    run_info["failure_mode"] = (
                        "full_load_converged" if fac == n_incr - 1 else "early_stop_target"
                    )
                    return _pack_return(best_dofs_all)

            # Divergence spike → rollback and stop
            if prev_res is not None and res_rms > growth_tol * prev_res:
                print(
                    f"[NH-guard] residual jump from {prev_res:.3e} to {res_rms:.3e} "
                    f"(>{growth_tol}×) at step {fac+1}, it {it}. "
                    f"Rollback to best and stop."
                )
                return _guard_return_current_or_best(fac, "growth_guard")

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
                return _guard_return_current_or_best(fac, "nonfinite_delta")

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
                                f"[NH-plateau] near-flat accepted step at step {fac+1}, it {it}: "
                                f"trial_rms={trial_rms:.3e}, prev_rms={res_rms:.3e}, streak={plateau_streak}."
                            )
                            if plateau_streak >= int(max(1, plateau_max_consecutive)):
                                print(
                                    f"[NH-guard] plateau stagnation at step {fac+1}, it {it}. "
                                    f"Reached {plateau_streak} consecutive near-flat accepted step(s)."
                                )
                                return _guard_return_current_or_best(fac, "plateau_stagnation")

                        if bt > 0:
                            print(
                                f"[NH-ls] accepted damped step at step {fac+1}, it {it}: "
                                f"alpha={alpha:.3e}, trial_rms={trial_rms:.3e}, prev_rms={res_rms:.3e}."
                            )
                        break

                    if bt < max_backtracks:
                        alpha *= shrink

                if not accepted:
                    print(
                        f"[NH-guard] line-search failed at step {fac+1}, it {it}. "
                        f"No acceptable alpha in {tried_backtracks + 1} trial(s); "
                        f"last_trial_rms={last_trial_rms:.3e}, prev_rms={res_rms:.3e}. "
                        f"Rollback to best and stop."
                    )
                    return _guard_return_current_or_best(fac, "line_search_failed")
            else:
                applied_dofs = applied_dofs.at[nd].add(delta[nd])

        else:
            return _guard_return_current_or_best(fac, "maxiter_reached")

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
        return _pack_return(best_dofs_full)

    if np.isfinite(best_res_all):
        print(
            f"[NH-done] finished all steps. "
            f"Best(overall) rms={best_res_all:.3e} at step {best_step_all[0]}, it {best_step_all[1]}."
        )
        run_info["fe_converged"] = False
        run_info["reached_full_load"] = False
        run_info["failed_load_step"] = n_incr
        run_info["reached_load_factor"] = float(best_step_all[0] / n_incr) if best_step_all[0] else 0.0
        run_info["failure_mode"] = "maxiter_reached"
        return _pack_return(best_dofs_all)

    print("[NH-done] no valid best state recorded; returning latest DOFs.")
    run_info["fe_converged"] = False
    run_info["reached_full_load"] = False
    run_info["failed_load_step"] = n_incr
    run_info["reached_load_factor"] = 0.0
    run_info["failure_mode"] = "no_valid_state"
    return _pack_return(applied_dofs)
