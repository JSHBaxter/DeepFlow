"""Compare ordinary binary MAP with a grid-aligned star-convexity prior.

Run from the repository root, using the environment containing the compiled
DeepFlow extension and matplotlib:

    python pytorch/example_star_convexity.py --output star_convexity.png

The synthetic data favours a ring with a hole and a small detached island.
The star prior requires foreground membership to be non-increasing away from
the chosen centre along both grid axes. These axis-wise constraints are a
discrete, grid-aligned prior, rather than arbitrary Euclidean ray constraints.
The centre is a direction-field reference; selecting it does not force it to
be foreground. Here its positive data score encourages foreground there.
"""

import argparse
from pathlib import Path

import torch
from binary_deepflow import BinaryDeepFlow2d, BinaryDeepFlow2dStar


def make_problem(size, device):
    centre = size // 2
    rows, columns = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    radius = torch.sqrt((rows - centre) ** 2 + (columns - centre) ** 2)

    # Positive scores favour label 1 (foreground), negative scores label 0.
    scores = torch.full_like(rows, -0.9)
    scores[(radius > size * 0.10) & (radius < size * 0.28)] = 0.7
    scores[radius < size * 0.03] = 3.0
    island = (rows - size * 0.82) ** 2 + (columns - size * 0.82) ** 2
    scores[island < (size * 0.06) ** 2] = 0.8

    # Every solver input has shape (batch, channels, rows, columns), float32.
    data = scores[None, None].contiguous()
    rx = torch.full_like(data, 0.15)  # Edges along dimension 2 (rows).
    ry = torch.full_like(data, 0.15)  # Edges along dimension 3 (columns).
    rx[:, :, -1, :] = 0  # No outgoing edge at the last row/column.
    ry[:, :, :, -1] = 0

    # Directions live on EDGES, using edge midpoints relative to the centre.
    # l = +1: u[next] <= u[current]; l = -1: u[next] >= u[current].
    # l = 0 disables the directed constraint on that edge.
    lx = torch.zeros_like(data)
    ly = torch.zeros_like(data)
    lx[:, :, :-1, :] = torch.sign(rows[:-1, :] + 0.5 - centre)
    ly[:, :, :, :-1] = torch.sign(columns[:, :-1] + 0.5 - centre)
    
    print(f"Data shape: {data.shape}, centre: {centre}, ")
    print(f"rx shape: {rx.shape}, ry shape: {ry.shape}, ")
    print(f"lx shape: {lx.shape}, ly shape: {ly.shape}, ")
    print(data)
    print(rx)
    print(lx)
    return data, rx, ry, lx, ly, centre


def constraint_violations(mask, lx, ly):
    row_change = (mask[:, :, 1:, :] - mask[:, :, :-1, :]) * lx[:, :, :-1, :]
    column_change = (mask[:, :, :, 1:] - mask[:, :, :, :-1]) * ly[:, :, :, :-1]
    return int((row_change > 0).sum() + (column_change > 0).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=65, help="Image side length (at least 17).")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("star_convexity.png"))
    args = parser.parse_args()
    if args.size < 17:
        parser.error("--size must be at least 17")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; run with --device cpu")
        import deepflow

        if not hasattr(deepflow, "binary_auglag_2dstar_gpu_forward"):
            parser.error("The DeepFlow extension was built without CUDA support")

    data, rx, ry, lx, ly, centre = make_problem(args.size, args.device)

    # eval() selects MAP inference. A newly constructed module defaults to
    # training mode, where the star module instead returns LBP marginals.
    ordinary_solver = BinaryDeepFlow2d(test_type="MAP").eval()
    star_solver = BinaryDeepFlow2dStar(test_type="MAP").eval()
    with torch.no_grad():
        ordinary_mask = ordinary_solver(data, rx, ry)
        star_mask = star_solver(data, rx, ry, lx, ly)

    for name, mask in (("Ordinary MAP", ordinary_mask), ("Star-constrained MAP", star_mask)):
        print(f"{name}: {int(mask.sum())} foreground pixels; "
              f"{constraint_violations(mask, lx, ly)} directed constraint violations")

    # Save a figure without opening a GUI, so this also works on a server.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    score_image = axes[0].imshow(data[0, 0].cpu(), cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_title("Data scores (+ favours foreground)")
    figure.colorbar(score_image, ax=axes[0], shrink=0.7)
    for axis, mask, title in (
        (axes[1], ordinary_mask, "Ordinary binary MAP"),
        (axes[2], star_mask, "Star-constrained binary MAP"),
    ):
        axis.imshow(mask[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
    for axis in axes:
        axis.plot(centre, centre, "+", color="lime", markersize=10)
        axis.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Saved comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
