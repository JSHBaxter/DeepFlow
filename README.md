Copyright (c) 2019, John S.H. Baxter
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL JOHN S.H. BAXTER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# DeepFlow
Continuous Max-Flow Layer in Tensorflow

This project includes C++/CUDA implementations of two different CRF solution
algorithms:
    (1) Marginal probability estimation via iterative mean-field approximations (MFI)
        which happens to be well-behaved and differentiable provided that the
        regularisation terms are kept within a particular bound, and
    (2) Maximum a posteriori probability estimation via an augmented Lagrangian
        solution algorithm (AL) [2], which is not differentiable but can be used for
        layer evaluation (but not for training). This solver works best if the data
        terms are normalized into a -1 to 1 range, but is mathematically scale-
        invariant.
    (3) Marginal probability estimation via loopy belief propagation (LBP) which
        is more well-behaved and differentable that the MFI solver. Yet to be published
        but you can find the pre-print on arXiv.

The MFI solver is based on that presented in 
[1] Baxter, J.S.H., Jannin, P. (2020) Topology-aware activation layer for neural network image segmentation. Proceedings of SPIE Medical Imaging, vol. 11313 p. 11313A. 

The AL solver is based on that presented in:
[2] Baxter, J.S.H., Rajchl, M., Yuan, J., & Peters, T. M. (2014). A continuous max-flow approach to multi-labeling problems under arbitrary region regularization. arXiv preprint arXiv:1405.0892.

At the moment, the project is designed for Linux and make. Depending on the
NVCC compiler used for the CUDA portions, one may need to set the corresponding
C compiler to a particular version using the C make variable.

    
--- NOTE LOG ---
    
(Please note that the TensorFlow portions are no longer being supported as we are
moving to a more research-friend PyTorch-native philosophy for new algorithm
development. Your understanding is appreciated.)
## PyTorch soft binary star LBP

```python
from binary_deepflow import BinaryDeepFlow2dStar

star = BinaryDeepFlow2dStar(
    test_type=None,          # Soft LBP in training and evaluation.
    check_every=100,         # First step, multiples of k, and the exact original cap.
    compile_updates=False, # Opt-in compiled tensor update blocks.
    record_history=False,  # Default: final scalar summaries only.
)
u, joint_rows, joint_columns = star(scores, rx, ry, lx, ly)
loss = dice_loss(u, targets)
loss.backward()
print(star.last_convergence["forward"]["status"])
```

Successful solver and precision-reference calls write nothing to stdout or stderr.
There is no progress-printing or callback policy. Default `test_type="MAP"` still
selects native hard MAP in evaluation; `test_type=None` keeps soft LBP in evaluation.
Native MAP behavior and ordinary LBP callers' original check schedule are unchanged.

`BinaryLBPOptions` contains only `check_every`, `compile_updates`, and `record_history`.
Use keyword construction. Update equations, arithmetic order, damping, epsilon values,
residual reductions and iteration limits are preserved. Buffers inherit the input
dtype, supporting float32 and float64 soft LBP. Compilation remains opt-in with
`fullgraph=True`, `dynamic=False` and explicit failures. First use includes compiler
overhead; compiled/eager agreement is not a mathematical accuracy guarantee.

### Final summaries and explicit history

`last_convergence` holds scalar-only forward/backward summaries: `phase`, `status`,
`converged`, `iterations`, `iteration_limit`, `tolerance`, `initial_residual`, `residual`,
and the existing message/gradient dtype fields. Status is `not_run` before a phase,
`running` at intermediate checks, and `converged`, `iteration_limit`, `nonfinite`, or
`zero_gradient` at termination. A cap-only exit is never reported as convergence.
Non-finite residuals raise at a scheduled check.

Pass `record_history=True` for plots. Each checked row adds iteration, residual,
mean/max absolute error and relative residual. Without it, no history rows,
history-only mean/max reductions or message/probability snapshots are allocated.
Checked scalars share one transfer. Forward retains the maximum sum of absolute
message errors; backward retains the maximum absolute message-gradient residual.
History mode does not change these stopping rules.

Each invocation owns its dictionary: a pending backward updates its own forward's
trace, not a newer call's trace. Save/copy a trace before another forward if needed.
No tensors are retained in the diagnostic dictionaries.
