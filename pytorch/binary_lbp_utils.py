"""Execution controls for binary LBP; tensor update equations match the eager solver.

Compilation is opt-in: fusion may introduce floating-point rounding differences.
Less frequent checks can delay stopping; damping and tolerances remain unchanged.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import torch


@dataclass(frozen=True)
class BinaryLBPOptions:
    check_every: int = 100
    compile_updates: bool = False
    record_history: bool = False

    def __post_init__(self):
        if isinstance(self.check_every, bool) or not isinstance(self.check_every, int) or self.check_every < 1:
            raise ValueError("check_every must be a positive integer")


def checked_iteration(iteration, limit, options):
    # Check the first step and exact legacy cap even when not a multiple of k.
    return iteration == 1 or iteration % options.check_every == 0 or iteration == limit


def peak_abs(values):
    # Python max(list_of_cuda_scalars) would synchronize for each comparison.
    return torch.stack([value.detach().abs().amax() for value in values]).amax()


def check_residual(values, iteration, limit, tolerance, phase, options, diagnostics):
    """Check the original residual criterion; optionally retain scalar history.

    Summary-only calls transfer one scalar. Explicit history mode groups residual,
    mean and maximum in one transfer, without retaining any tensor in diagnostics.
    """
    absolute = [value.detach().abs() for value in values]
    if phase == "forward":
        residual = torch.stack([value.sum() for value in absolute]).amax()
    else:
        residual = torch.stack([value.amax() for value in absolute]).amax()
    keep_history = diagnostics is not None and options.record_history
    if keep_history:
        mean = torch.stack([value.mean() for value in absolute]).amax()
        peak = torch.stack([value.amax() for value in absolute]).amax()
        residual, mean, peak = torch.stack((residual, mean, peak)).cpu().tolist()
        finite = all(math.isfinite(value) for value in (residual, mean, peak))
    else:
        residual = residual.item()
        finite = math.isfinite(residual)
    converged = finite and (residual < tolerance if phase == "forward" else residual <= tolerance)
    status = ("nonfinite" if not finite else "converged" if converged
              else "iteration_limit" if iteration == limit else "running")
    if diagnostics is not None:
        if diagnostics.get("initial_residual") is None:
            diagnostics["initial_residual"] = residual
        if keep_history:
            initial = diagnostics["initial_residual"]
            diagnostics.setdefault("history", []).append(dict(
                iteration=iteration, residual=residual, mean_abs=mean, max_abs=peak,
                relative_residual=residual / initial if initial else 0.0,
            ))
        diagnostics.update(phase=phase, iterations=iteration, iteration_limit=limit,
                           tolerance=tolerance, residual=residual, converged=converged, status=status)
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


def forward_step(vertex_messages, ed0, ed1, eru, erd, meu0, med0, meu1, med1, compute_error):
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
    return meu0, med0, meu1, med1, errors


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

