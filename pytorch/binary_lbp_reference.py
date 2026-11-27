"""Frozen-input float32/float64 reference for the soft binary star LBP solver.

The U-Net is deliberately outside this comparison. Identical unary scores, finite
weights, directions, targets, equations and stopping tolerances are used in both
runs. Both precisions use the original residual criteria and iteration limits.
An unconverged float64 run is explicitly reported and is not an accuracy certificate.
"""
import copy
import math
from time import perf_counter

import torch

from binary_deepflow import BinaryDeepFlow2dStar


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compare_binary_star_precision(inputs, targets, *, check_every=100,
                                  compile_updates=False,
                                  smooth=1e-6, metadata=None):
    """Return (JSON-compatible report, CPU tensor artifacts) without optimizer steps.

    ``inputs`` is (scores, row_weights, column_weights, row_directions,
    column_directions), all float32 [B,1,H,W] on one device. Float64 casts exactly the same
    stored problem values as float32; it does not recompute U-Net scores or weights.
    Gradients are of mean soft Dice loss with respect to the fixed unary scores.
    Timings include any first-use compilation. Artifacts preserve the frozen problem
    and both outputs/gradients for independent inspection or later reruns.
    """
    if len(inputs) != 5:
        raise ValueError("Expected (scores, rx, ry, lx, ly)")
    scores = inputs[0]
    if scores.ndim != 4 or scores.shape[1] != 1 or min(scores.shape) < 1:
        raise ValueError("Expected nonempty scores [B,1,H,W]")
    if min(scores.shape[-2:]) < 2:
        raise ValueError("Both spatial dimensions must be at least 2")
    if scores.dtype != torch.float32:
        raise TypeError("Freeze inputs in float32 first; float64 then uses identical stored values")
    if any(t.shape != scores.shape or t.device != scores.device or t.dtype != scores.dtype for t in inputs):
        raise ValueError("All five inputs must share shape, device and dtype")
    if targets.shape != scores.shape:
        raise ValueError("Targets and scores must have matching shapes")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("Dice smoothing must be positive")
    if not all(torch.isfinite(t).all() for t in inputs):
        raise ValueError("All frozen inputs must be finite; hard penalties are built by the layer")
    if (inputs[1] < 0).any() or (inputs[2] < 0).any():
        raise ValueError("Finite pairwise weights must be nonnegative")
    if any(not ((direction == -1) | (direction == 0) | (direction == 1)).all()
           for direction in inputs[3:]):
        raise ValueError("Directions must be -1, 0 or 1")
    if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("Targets must be finite and in [0,1]")

    # No aliasing with caller inputs, and no connection to its U-Net graph.
    frozen = tuple(t.detach().clone() for t in inputs)
    target = targets.detach().to(scores.device).clone()
    report = dict(
        shape=list(scores.shape), device=str(scores.device), torch_version=str(torch.__version__),
        source_dtype=str(scores.dtype), compile_updates=compile_updates,
        check_every=check_every, smooth=smooth,
        metadata=copy.deepcopy(metadata or {}),
        timing_note="Wall time includes first-use compilation; not a steady-state benchmark.",
    )
    artifacts = {"inputs": dict(zip(("scores", "rx", "ry", "lx", "ly"),
                                    (t.cpu() for t in frozen))), "targets": target.cpu()}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        solver = BinaryDeepFlow2dStar(
            test_type=None, check_every=check_every, compile_updates=compile_updates,
            record_history=True,
        )
        d = frozen[0].to(dtype=dtype).clone().requires_grad_()
        rest = tuple(t.to(dtype=dtype).contiguous() for t in frozen[1:])
        y = target.to(dtype=dtype)
        with torch.enable_grad():
            _synchronize(d.device)
            started = perf_counter()
            probabilities = solver(d, *rest)[0]
            _synchronize(d.device)
            forward_seconds = perf_counter() - started
            if probabilities.dtype != dtype:
                raise AssertionError(f"Solver returned {probabilities.dtype}; expected {dtype}")
            if solver.last_convergence["forward"].get("message_dtype") != str(dtype):
                raise RuntimeError("Restart the kernel to load the dtype-preserving LBP message buffers")
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError(f"Non-finite {name} probabilities")
            intersection = (probabilities * y).sum((1, 2, 3))
            soft_dice = ((2 * intersection + smooth) /
                         (probabilities.sum((1, 2, 3)) + y.sum((1, 2, 3)) + smooth)).mean()
            loss = 1 - soft_dice
            _synchronize(d.device)
            started = perf_counter()
            gradient = torch.autograd.grad(loss, d)[0]
            _synchronize(d.device)
            backward_seconds = perf_counter() - started
            if gradient.dtype != dtype or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Invalid {name} unary gradient")
            if solver.last_convergence["backward"].get("gradient_dtype") != str(dtype):
                raise AssertionError("LBP gradient buffer dtype does not match the reference input")
        hard = (probabilities.detach() >= .5).to(dtype)
        hard_dice = ((2 * (hard*y).sum((1, 2, 3)) + smooth) /
                     (hard.sum((1, 2, 3)) + y.sum((1, 2, 3)) + smooth)).mean()
        trace = copy.deepcopy(solver.last_convergence)
        report[name] = dict(dtype=str(dtype), loss=loss.item(), soft_dice=soft_dice.item(),
                            hard_dice=hard_dice.item(), forward_seconds=forward_seconds,
                            backward_seconds=backward_seconds, trace=trace)
        artifacts[name] = dict(probabilities=probabilities.detach().cpu(),
                               unary_gradient=gradient.detach().cpu())


    # Compare in float64 on CPU so subtraction/comparison adds no float32 rounding.
    p32 = artifacts["float32"]["probabilities"].double()
    p64 = artifacts["float64"]["probabilities"]
    g32 = artifacts["float32"]["unary_gradient"].double().flatten()
    g64 = artifacts["float64"]["unary_gradient"].flatten()
    probability_error = (p32 - p64).abs()
    gradient_error = (g32 - g64).norm().item()
    norm32, norm64 = g32.norm().item(), g64.norm().item()
    report["differences"] = dict(
        probability_max_abs=probability_error.max().item(),
        probability_mean_abs=probability_error.mean().item(),
        mask_disagreement_fraction=((p32 >= .5) != (p64 >= .5)).double().mean().item(),
        loss_float32_minus_float64=report["float32"]["loss"]-report["float64"]["loss"],
        soft_dice_float32_minus_float64=report["float32"]["soft_dice"]-report["float64"]["soft_dice"],
        hard_dice_float32_minus_float64=report["float32"]["hard_dice"]-report["float64"]["hard_dice"],
        gradient_difference_l2=gradient_error,
        gradient_relative_l2_vs_float64=gradient_error/norm64 if norm64 > 0 else None,
        gradient_cosine=(torch.dot(g32/norm32, g64/norm64).item()
                         if norm32 > 0 and norm64 > 0 else None),
    )
    report["float64_converged"] = all(
        report["float64"]["trace"][phase]["converged"] for phase in ("forward", "backward"))
    return report, artifacts
