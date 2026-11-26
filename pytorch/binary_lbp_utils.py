"""Execution controls for binary LBP; tensor update equations match the eager solver.

Compilation is opt-in: fusion may introduce floating-point rounding differences.
Checks less often can delay stopping; damping and tolerances remain unchanged.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import torch


@dataclass(frozen=True)
class BinaryLBPOptions:
    check_every: int = 100
    compile_updates: bool = False
    record_history: bool = True
    report_every: int = 0
    monitor_updates: bool = True
    stop_on_stall: bool = False
    stall_patience: int = 5  # Consecutive checked intervals, not iterations.
    stall_min_iterations: int = 1000
    stall_message_atol: float = 0.0
    stall_probability_atol: float = 0.0

    def __post_init__(self):
        for name, minimum in (("check_every", 1), ("report_every", 0),
                              ("stall_patience", 1), ("stall_min_iterations", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("stall_message_atol", "stall_probability_atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.stop_on_stall and not self.monitor_updates:
            raise ValueError("stop_on_stall=True requires monitor_updates=True")


def checked_iteration(iteration, limit, options):
    # Check the first step and exact legacy cap even when not a multiple of k.
    return iteration == 1 or iteration % options.check_every == 0 or iteration == limit


def peak_abs(values):
    # Python max(list_of_cuda_scalars) would synchronize for each comparison.
    return torch.stack([value.detach().abs().amax() for value in values]).amax()


def check_residual(values, iteration, limit, tolerance, phase, options, diagnostics,
                   forward_metrics=None):
    absolute = [value.detach().abs() for value in values]
    if phase == "forward":
        residual = torch.stack([value.sum() for value in absolute]).amax()
    else:
        residual = torch.stack([value.amax() for value in absolute]).amax()
    forward_metrics = forward_metrics or {}
    names = [name for name, value in forward_metrics.items() if value is not None]
    if diagnostics is not None:
        mean = torch.stack([value.mean() for value in absolute]).amax()
        peak = torch.stack([value.amax() for value in absolute]).amax()
        # One transfer includes all diagnostics; no additional host decisions.
        scalars = torch.stack((residual, mean, peak, *(forward_metrics[name] for name in names))).cpu().tolist()
        residual, mean, peak = scalars[:3]
        metrics = dict.fromkeys(forward_metrics)
        metrics.update(zip(names, scalars[3:]))
        finite = all(math.isfinite(value) for value in scalars)
    else:
        residual, mean, peak = residual.item(), None, None
        metrics = {}
        finite = math.isfinite(residual)
    converged = finite and (residual < tolerance if phase == "forward" else residual <= tolerance)
    stagnant = False
    stable_checks = 0
    if phase == "forward" and diagnostics is not None and metrics:
        # Require both the instantaneous update and the drift since the previous
        # check to be small, plus probability stability. The first check has no
        # previous snapshot and can never count as a stable interval.
        stable = finite and iteration >= options.stall_min_iterations
        for name, atol in (("actual_update_max_abs", options.stall_message_atol),
                           ("message_change_max_abs", options.stall_message_atol),
                           ("probability_change_max_abs", options.stall_probability_atol)):
            value = metrics.get(name)
            stable = stable and value is not None and value <= atol
        stable_checks = diagnostics.get("stable_checks", 0) + 1 if stable else 0
        stagnant = stable_checks >= options.stall_patience
    status = ("nonfinite" if not finite else "converged" if converged
              else "stalled" if stagnant and options.stop_on_stall
              else "iteration_limit" if iteration == limit else "running")
    if diagnostics is not None:
        initial = diagnostics.setdefault("initial_residual", residual)
        row = dict(iteration=iteration, residual=residual, mean_abs=mean, max_abs=peak,
                   relative_residual=residual / initial if initial else 0.0, **metrics)
        if phase == "forward" and metrics:
            reason = ("exact_freeze" if options.stall_message_atol == 0 and options.stall_probability_atol == 0
                      else "within_tolerance") if stagnant else None
            row.update(stable_checks=stable_checks, stagnation_detected=stagnant)
            diagnostics.update(stable_checks=stable_checks, stagnation_detected=stagnant,
                               stagnation_reason=reason, **metrics)
        if options.record_history:
            diagnostics.setdefault("history", []).append(row)
        diagnostics.update(phase=phase, iterations=iteration, iteration_limit=limit,
                           tolerance=tolerance, residual=residual, converged=converged, status=status)
    if options.report_every:
        if diagnostics is not None:
            due = iteration - diagnostics.get("last_report_iteration", 0) >= options.report_every
        else:
            due = iteration % options.report_every == 0
        if iteration == 1 or due or status != "running":
            extra = ""
            if metrics:
                probability_change = metrics.get("probability_change_max_abs")
                probability_text = "n/a" if probability_change is None else f"{probability_change:.3g}"
                extra = (f", update_max={metrics['actual_update_max_abs']:.3g}, "
                         f"message_delta={metrics['message_change_max_abs'] if metrics['message_change_max_abs'] is not None else 'n/a'}, "
                         f"prob_delta={probability_text}, stagnant={stagnant}")
            print(f"LBP {phase}: {iteration}/{limit}, residual={residual:.6g}, "
                  f"tol={tolerance:.3g}, status={status}{extra}", flush=True)
            if diagnostics is not None:
                diagnostics["last_report_iteration"] = iteration
    if not finite:
        raise FloatingPointError(f"Non-finite binary LBP {phase} residual at iteration {iteration}")
    return status != "running"


@lru_cache(maxsize=2)
def compiled_step(phase):
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_updates=True requires torch.compile; use False for eager updates")
    step = forward_step if phase == "forward" else backward_step
    # No convergence loop is compiled/unrolled. Fullgraph makes failures explicit.
    return torch.compile(step, fullgraph=True, dynamic=False)


def forward_step(vertex_messages, ed0, ed1, eru, erd, meu0, med0, meu1, med1,
                 compute_error, monitor_updates=False):
    dims, alpha, m_epsilon = len(eru), 0.05, 1e-10
    err_med0, err_med1, err_meu0, err_meu1 = ([None] * dims for _ in range(4))
    mvu0, mvd0, mvu1, mvd1 = vertex_messages(ed0, ed1, meu0, med0, meu1, med1)

    #normalised messages
    avu = [mvu0[i]+mvu1[i]+2*m_epsilon for i in range(dims)]
    avd = [mvd0[i]+mvd1[i]+2*m_epsilon for i in range(dims)]
    for i in range(dims):
        mvu0[i] = (mvu0[i]+m_epsilon) / avu[i]
        mvu1[i] = (mvu1[i]+m_epsilon) / avu[i]
        mvd0[i] = (mvd0[i]+m_epsilon) / avd[i]
        mvd1[i] = (mvd1[i]+m_epsilon) / avd[i]

    # Retain old tensor references before rebinding the message lists. These
    # tensors are not modified by this update; no clones are needed in the kernel.
    before = med0 + med1 + meu0 + meu1 if monitor_updates else []
    #compute edge to vertex messages
    for i in range(dims):
        if i == 0:
            new_med0 = mvd0[i] + eru[i][:,:,:-1,...] * mvd1[i]
            new_med1 = mvd1[i] + erd[i][:,:,:-1,...] * mvd0[i]
            new_meu0 = mvu0[i] + erd[i][:,:,:-1,...] * mvu1[i]
            new_meu1 = mvu1[i] + eru[i][:,:,:-1,...] * mvu0[i]
        elif i == 1:
            new_med0 = mvd0[i] + eru[i][:,:,:,:-1,...] * mvd1[i]
            new_med1 = mvd1[i] + erd[i][:,:,:,:-1,...] * mvd0[i]
            new_meu0 = mvu0[i] + erd[i][:,:,:,:-1,...] * mvu1[i]
            new_meu1 = mvu1[i] + eru[i][:,:,:,:-1,...] * mvu0[i]
        elif i == 2:
            new_med0 = mvd0[i] + eru[i][:,:,:,:,:-1,...] * mvd1[i]
            new_med1 = mvd1[i] + erd[i][:,:,:,:,:-1,...] * mvd0[i]
            new_meu0 = mvu0[i] + erd[i][:,:,:,:,:-1,...] * mvu1[i]
            new_meu1 = mvu1[i] + eru[i][:,:,:,:,:-1,...] * mvu0[i]
        if compute_error:
            err_med0[i] = med0[i] - new_med0
            err_med1[i] = med1[i] - new_med1
            err_meu0[i] = meu0[i] - new_meu0
            err_meu1[i] = meu1[i] - new_meu1
        med0[i] = (1-alpha)*med0[i] + alpha*new_med0
        med1[i] = (1-alpha)*med1[i] + alpha*new_med1
        meu0[i] = (1-alpha)*meu0[i] + alpha*new_meu0
        meu1[i] = (1-alpha)*meu1[i] + alpha*new_meu1

    errors = err_med0 + err_med1 + err_meu0 + err_meu1 if compute_error else []
    updates = [new - old for new, old in zip(med0 + med1 + meu0 + meu1, before)]
    return meu0, med0, meu1, med1, errors, updates


def node_probabilities(ed0, ed1, meu0, med0, meu1, med1):
    """Same foreground marginal arithmetic/order as the binary layer, for 1D–3D."""
    from torch.nn.functional import pad
    dims = len(meu0)
    node0, node1 = ed0, ed1
    for axis in range(dims):
        down, up = [0] * (2 * dims), [0] * (2 * dims)
        down[2 * (dims - axis - 1) + 1] = 1
        up[2 * (dims - axis - 1)] = 1
        node0 = node0 * pad(med0[axis], tuple(down), value=1)
        node0 = node0 * pad(meu0[axis], tuple(up), value=1)
        node1 = node1 * pad(med1[axis], tuple(down), value=1)
        node1 = node1 * pad(meu1[axis], tuple(up), value=1)
    return node1 / (node0 + node1)


def forward_change_metrics(updates, messages, probabilities, previous_messages, previous_probabilities):
    """Scalar tensors only. Differences span one step or one checked interval."""
    absolute_updates = [update.abs() for update in updates]
    metrics = {
        "actual_update_max_abs": torch.stack([value.amax() for value in absolute_updates]).amax(),
        "actual_update_mean_abs": torch.stack([value.mean() for value in absolute_updates]).amax(),
        "message_change_max_abs": None,
        "probability_change_max_abs": None,
        "probability_change_mean_abs": None,
        # Also detects invalid probabilities at the first check, before deltas exist.
        "probability_min": probabilities.amin(),
        "probability_max": probabilities.amax(),
    }
    if previous_messages is not None:
        metrics["message_change_max_abs"] = peak_abs(
            [new - old for new, old in zip(messages, previous_messages)])
        change = (probabilities - previous_probabilities).abs()
        metrics["probability_change_max_abs"] = change.amax()
        metrics["probability_change_mean_abs"] = change.mean()
    return metrics


def backward_step(grad_meu0, grad_med0, grad_meu1, grad_med1,
                  grad_d0, grad_d1, grad_ru, grad_rd, d, eru, erd,
                  mvu0, mvd0, mvu1, mvd1, nmvu0, nmvd0, nmvu1, nmvd1,
                  dnmvu0, dnmvd0, dnmvu1, dnmvd1, meu0, med0, meu1, med1):
    dims = len(eru)
    grad_mvu0, grad_mvd0, grad_mvu1, grad_mvd1 = ([None] * dims for _ in range(4))
    #backprop back to vertex messages

    for i in range(dims):
        if i == 0:
            grad_mvu0[i] = grad_meu0[i] + eru[i][:,:,:-1,...]*grad_meu1[i]
            grad_mvu1[i] = grad_meu1[i] + erd[i][:,:,:-1,...]*grad_meu0[i]
            grad_mvd0[i] = grad_med0[i] + erd[i][:,:,:-1,...]*grad_med1[i]
            grad_mvd1[i] = grad_med1[i] + eru[i][:,:,:-1,...]*grad_med0[i]
            grad_ru[i][:,:,:-1,...] += grad_meu1[i]*nmvu0[i] + grad_meu0[i]*nmvu1[i]
            grad_rd[i][:,:,:-1,...] += grad_med1[i]*nmvd0[i] + grad_med0[i]*nmvd1[i]
        if i == 1:
            grad_mvu0[i] = grad_meu0[i] + eru[i][:,:,:,:-1,...]*grad_meu1[i]
            grad_mvu1[i] = grad_meu1[i] + erd[i][:,:,:,:-1,...]*grad_meu0[i]
            grad_mvd0[i] = grad_med0[i] + erd[i][:,:,:,:-1,...]*grad_med1[i]
            grad_mvd1[i] = grad_med1[i] + eru[i][:,:,:,:-1,...]*grad_med0[i]
            grad_ru[i][:,:,:,:-1,...] += grad_meu1[i]*nmvu0[i] + grad_meu0[i]*nmvu1[i]
            grad_rd[i][:,:,:,:-1,...] += grad_med1[i]*nmvd0[i] + grad_med0[i]*nmvd1[i]
        if i == 2:
            grad_mvu0[i] = grad_meu0[i] + eru[i][:,:,:,:,:-1,...]*grad_meu1[i]
            grad_mvu1[i] = grad_meu1[i] + erd[i][:,:,:,:,:-1,...]*grad_meu0[i]
            grad_mvd0[i] = grad_med0[i] + erd[i][:,:,:,:,:-1,...]*grad_med1[i]
            grad_mvd1[i] = grad_med1[i] + eru[i][:,:,:,:,:-1,...]*grad_med0[i]
            grad_ru[i][:,:,:,:,:-1,...] += grad_meu1[i]*nmvu0[i] + grad_meu0[i]*nmvu1[i]
            grad_rd[i][:,:,:,:,:-1,...] += grad_med1[i]*nmvd0[i] + grad_med0[i]*nmvd1[i]


    #backprop through renormalisation
    for i in range(dims):
        grad_mvu0[i], grad_mvu1[i] = dnmvu1[i]*(grad_mvu0[i] - grad_mvu1[i]), dnmvu0[i]*(grad_mvu1[i] - grad_mvu0[i])
        grad_mvd0[i], grad_mvd1[i] = dnmvd1[i]*(grad_mvd0[i] - grad_mvd1[i]), dnmvd0[i]*(grad_mvd1[i] - grad_mvd0[i])

    #clear gradient buffers for the edge messages (about to compute updated ones
    alpha = 0.25 #dampening rate
    for i in range(dims):
        grad_med0[i] *= (1-alpha)/alpha
        grad_meu0[i] *= (1-alpha)/alpha
        grad_med1[i] *= (1-alpha)/alpha
        grad_meu1[i] *= (1-alpha)/alpha

    #handle data terms
    for ni in range(dims):
        #print("grad_mvu1",grad_mvu1[ni])
        #print("mvu1",mvu1[ni])
        if ni == 0:
            grad_d1[:,:,:-1,...] += (grad_mvu1[ni] * mvu1[ni])*(d[:,:,:-1,...]<0)
            grad_d1[:,:,1:,...] += (grad_mvd1[ni] * mvd1[ni])*(d[:,:,1:,...]<0)
            grad_d0[:,:,:-1,...] += (grad_mvu0[ni] * mvu0[ni])*(d[:,:,:-1,...]>0)
            grad_d0[:,:,1:,...] += (grad_mvd0[ni] * mvd0[ni])*(d[:,:,1:,...]>0)
        elif ni == 1:
            grad_d1[:,:,:,:-1,...] += (grad_mvu1[ni] * mvu1[ni])*(d[:,:,:,:-1,...]<0)
            grad_d1[:,:,:,1:,...] += (grad_mvd1[ni] * mvd1[ni])*(d[:,:,:,1:,...]<0)
            grad_d0[:,:,:,:-1,...] += (grad_mvu0[ni] * mvu0[ni])*(d[:,:,:,:-1,...]>0)
            grad_d0[:,:,:,1:,...] += (grad_mvd0[ni] * mvd0[ni])*(d[:,:,:,1:,...]>0)
        elif ni == 2:
            grad_d1[:,:,:,:,:-1,...] += (grad_mvu1[ni] * mvu1[ni])*(d[:,:,:,:,:-1,...]<0)
            grad_d1[:,:,:,:,1:,...] += (grad_mvd1[ni] * mvd1[ni])*(d[:,:,:,:,1:,...]<0)
            grad_d0[:,:,:,:,:-1,...] += (grad_mvu0[ni] * mvu0[ni])*(d[:,:,:,:,:-1,...]>0)
            grad_d0[:,:,:,:,1:,...] += (grad_mvd0[ni] * mvd0[ni])*(d[:,:,:,:,1:,...]>0)

    #handle colinear edges
    for ni in range(dims):
        if ni == 0:
            grad_meu1[ni][:,:,:-1,...] += grad_mvu1[ni][:,:,1:,...] * mvu1[ni][:,:,1:,...] / meu1[ni][:,:,:-1,...]
            grad_meu0[ni][:,:,:-1,...] += grad_mvu0[ni][:,:,1:,...] * mvu0[ni][:,:,1:,...] / meu0[ni][:,:,:-1,...]
            grad_med1[ni][:,:,1:,...] += grad_mvd1[ni][:,:,:-1,...] * mvd1[ni][:,:,:-1,...] / med1[ni][:,:,1:,...]
            grad_med0[ni][:,:,1:,...] += grad_mvd0[ni][:,:,:-1,...] * mvd0[ni][:,:,:-1,...] / med0[ni][:,:,1:,...]
        elif ni == 1:
            grad_meu1[ni][:,:,:,:-1,...] += grad_mvu1[ni][:,:,:,1:,...] * mvu1[ni][:,:,:,1:,...] / meu1[ni][:,:,:,:-1,...]
            grad_meu0[ni][:,:,:,:-1,...] += grad_mvu0[ni][:,:,:,1:,...] * mvu0[ni][:,:,:,1:,...] / meu0[ni][:,:,:,:-1,...]
            grad_med1[ni][:,:,:,1:,...] += grad_mvd1[ni][:,:,:,:-1,...] * mvd1[ni][:,:,:,:-1,...] / med1[ni][:,:,:,1:,...]
            grad_med0[ni][:,:,:,1:,...] += grad_mvd0[ni][:,:,:,:-1,...] * mvd0[ni][:,:,:,:-1,...] / med0[ni][:,:,:,1:,...]
        elif ni == 2:
            grad_meu1[ni][:,:,:,:,:-1,...] += grad_mvu1[ni][:,:,:,:,1:,...] * mvu1[ni][:,:,:,:,1:,...] / meu1[ni][:,:,:,:,:-1,...]
            grad_meu0[ni][:,:,:,:,:-1,...] += grad_mvu0[ni][:,:,:,:,1:,...] * mvu0[ni][:,:,:,:,1:,...] / meu0[ni][:,:,:,:,:-1,...]
            grad_med1[ni][:,:,:,:,1:,...] += grad_mvd1[ni][:,:,:,:,:-1,...] * mvd1[ni][:,:,:,:,:-1,...] / med1[ni][:,:,:,:,1:,...]
            grad_med0[ni][:,:,:,:,1:,...] += grad_mvd0[ni][:,:,:,:,:-1,...] * mvd0[ni][:,:,:,:,:-1,...] / med0[ni][:,:,:,:,1:,...]

    # Accumulate contributions from every source direction.
    for ni in range(dims):
        #put in contributions from other directions
        for di2 in range(dims):
            if ni == di2:
                continue

            if di2 == 0:
                grad_contribution_1_d_d = grad_mvd1[ni][:,:,:-1,...] * mvd1[ni][:,:,:-1,...]
                grad_contribution_1_u_d = grad_mvd1[ni][:,:,1:,...] * mvd1[ni][:,:,1:,...]
                grad_contribution_1_d_u = grad_mvu1[ni][:,:,:-1,...] * mvu1[ni][:,:,:-1,...]
                grad_contribution_1_u_u = grad_mvu1[ni][:,:,1:,...] * mvu1[ni][:,:,1:,...]
                grad_contribution_0_d_d = grad_mvd0[ni][:,:,:-1,...] * mvd0[ni][:,:,:-1,...]
                grad_contribution_0_u_d = grad_mvd0[ni][:,:,1:,...] * mvd0[ni][:,:,1:,...]
                grad_contribution_0_d_u = grad_mvu0[ni][:,:,:-1,...] * mvu0[ni][:,:,:-1,...]
                grad_contribution_0_u_u = grad_mvu0[ni][:,:,1:,...] * mvu0[ni][:,:,1:,...]
            elif di2 == 1:
                grad_contribution_1_d_d = grad_mvd1[ni][:,:,:,:-1,...] * mvd1[ni][:,:,:,:-1,...]
                grad_contribution_1_u_d = grad_mvd1[ni][:,:,:,1:,...] * mvd1[ni][:,:,:,1:,...]
                grad_contribution_1_d_u = grad_mvu1[ni][:,:,:,:-1,...] * mvu1[ni][:,:,:,:-1,...]
                grad_contribution_1_u_u = grad_mvu1[ni][:,:,:,1:,...] * mvu1[ni][:,:,:,1:,...]
                grad_contribution_0_d_d = grad_mvd0[ni][:,:,:,:-1,...] * mvd0[ni][:,:,:,:-1,...]
                grad_contribution_0_u_d = grad_mvd0[ni][:,:,:,1:,...] * mvd0[ni][:,:,:,1:,...]
                grad_contribution_0_d_u = grad_mvu0[ni][:,:,:,:-1,...] * mvu0[ni][:,:,:,:-1,...]
                grad_contribution_0_u_u = grad_mvu0[ni][:,:,:,1:,...] * mvu0[ni][:,:,:,1:,...]
            elif di2 == 2:
                grad_contribution_1_d_d = grad_mvd1[ni][:,:,:,:,:-1,...] * mvd1[ni][:,:,:,:,:-1,...]
                grad_contribution_1_u_d = grad_mvd1[ni][:,:,:,:,1:,...] * mvd1[ni][:,:,:,:,1:,...]
                grad_contribution_1_d_u = grad_mvu1[ni][:,:,:,:,:-1,...] * mvu1[ni][:,:,:,:,:-1,...]
                grad_contribution_1_u_u = grad_mvu1[ni][:,:,:,:,1:,...] * mvu1[ni][:,:,:,:,1:,...]
                grad_contribution_0_d_d = grad_mvd0[ni][:,:,:,:,:-1,...] * mvd0[ni][:,:,:,:,:-1,...]
                grad_contribution_0_u_d = grad_mvd0[ni][:,:,:,:,1:,...] * mvd0[ni][:,:,:,:,1:,...]
                grad_contribution_0_d_u = grad_mvu0[ni][:,:,:,:,:-1,...] * mvu0[ni][:,:,:,:,:-1,...]
                grad_contribution_0_u_u = grad_mvu0[ni][:,:,:,:,1:,...] * mvu0[ni][:,:,:,:,1:,...]

            if ni == 0:
                grad_med1[di2][:,:,:-1,...] += grad_contribution_1_d_u / med1[di2][:,:,:-1,...]
                grad_meu1[di2][:,:,:-1,...] += grad_contribution_1_u_u / meu1[di2][:,:,:-1,...]
                grad_med1[di2][:,:,1:,...] += grad_contribution_1_d_d / med1[di2][:,:,1:,...]
                grad_meu1[di2][:,:,1:,...] += grad_contribution_1_u_d / meu1[di2][:,:,1:,...]
                grad_med0[di2][:,:,:-1,...] += grad_contribution_0_d_u / med0[di2][:,:,:-1,...]
                grad_meu0[di2][:,:,:-1,...] += grad_contribution_0_u_u / meu0[di2][:,:,:-1,...]
                grad_med0[di2][:,:,1:,...] += grad_contribution_0_d_d / med0[di2][:,:,1:,...]
                grad_meu0[di2][:,:,1:,...] += grad_contribution_0_u_d / meu0[di2][:,:,1:,...]
            elif ni == 1:
                grad_med1[di2][:,:,:,:-1,...] += grad_contribution_1_d_u / med1[di2][:,:,:,:-1,...]
                grad_meu1[di2][:,:,:,:-1,...] += grad_contribution_1_u_u / meu1[di2][:,:,:,:-1,...]
                grad_med1[di2][:,:,:,1:,...] += grad_contribution_1_d_d / med1[di2][:,:,:,1:,...]
                grad_meu1[di2][:,:,:,1:,...] += grad_contribution_1_u_d / meu1[di2][:,:,:,1:,...]
                grad_med0[di2][:,:,:,:-1,...] += grad_contribution_0_d_u / med0[di2][:,:,:,:-1,...]
                grad_meu0[di2][:,:,:,:-1,...] += grad_contribution_0_u_u / meu0[di2][:,:,:,:-1,...]
                grad_med0[di2][:,:,:,1:,...] += grad_contribution_0_d_d / med0[di2][:,:,:,1:,...]
                grad_meu0[di2][:,:,:,1:,...] += grad_contribution_0_u_d / meu0[di2][:,:,:,1:,...]
            elif ni == 2:
                grad_med1[di2][:,:,:,:,:-1,...] += grad_contribution_1_d_u / med1[di2][:,:,:,:,:-1,...]
                grad_meu1[di2][:,:,:,:,:-1,...] += grad_contribution_1_u_u / meu1[di2][:,:,:,:,:-1,...]
                grad_med1[di2][:,:,:,:,1:,...] += grad_contribution_1_d_d / med1[di2][:,:,:,:,1:,...]
                grad_meu1[di2][:,:,:,:,1:,...] += grad_contribution_1_u_d / meu1[di2][:,:,:,:,1:,...]
                grad_med0[di2][:,:,:,:,:-1,...] += grad_contribution_0_d_u / med0[di2][:,:,:,:,:-1,...]
                grad_meu0[di2][:,:,:,:,:-1,...] += grad_contribution_0_u_u / meu0[di2][:,:,:,:,:-1,...]
                grad_med0[di2][:,:,:,:,1:,...] += grad_contribution_0_d_d / med0[di2][:,:,:,:,1:,...]
                grad_meu0[di2][:,:,:,:,1:,...] += grad_contribution_0_u_d / meu0[di2][:,:,:,:,1:,...]

    #correct for scaling
    for i in range(dims):
        grad_med0[i] *= alpha
        grad_meu0[i] *= alpha
        grad_med1[i] *= alpha
        grad_meu1[i] *= alpha

