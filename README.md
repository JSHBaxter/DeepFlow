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
## PyTorch star LBP execution controls

`BinaryDeepFlow2dStar` supports less frequent convergence checks and optional
compiled tensor updates in both forward and backward:

```python
from binary_deepflow import BinaryDeepFlow2dStar

star = BinaryDeepFlow2dStar(
    test_type=None,          # Soft LBP in training and evaluation.
    check_every=100,        # First iteration, every k steps, and the original cap.
    compile_updates=True,  # Optional torch.compile fusion; default is False.
    record_history=True,
    report_every=5000,     # Live progress at checked iterations; default is 0.
)
probabilities, joint_rows, joint_columns = star(scores, rx, ry, lx, ly)
loss = dice_loss(probabilities, targets)
loss.backward()
trace = star.last_convergence
print(trace["forward"]["status"], trace["backward"]["status"])
```

Each phase records its iterations, iteration limit, tolerance, final residual,
convergence status and scalar-only history. Status `iteration_limit` means the
solver exhausted its budget without satisfying the stopping criterion. History
rows include iteration, residual, mean/max absolute error and relative residual.
For forward, residual is the maximum **sum** of absolute message errors among
message tensors. Backward uses the maximum absolute message-gradient residual.
The diagnostic mean does not change either stopping rule. Non-finite residuals
raise an error at the next check.

Only the latest invocation is retained by the module; save or copy the dictionary
before another forward to keep a trace. A pending backward updates its own forward's
dictionary. The optimized segmentation notebook saves the smoke-test trace as JSON
and plots the curves. `record_history=False` keeps final summaries without history.

The original update equations, damping, tolerances and final iteration limits are
retained. Checks less often may delay early stopping by up to `check_every - 1`
iterations and consequently change the returned probabilities and gradients.
`check_every=1, compile_updates=False` selects the original schedule and eager
arithmetic. Ordinary LBP callers retain their original per-iteration check schedule.

Compilation requires a compatible PyTorch compiler installation and is opt-in.
It fuses update blocks, not the convergence loop, and does not guarantee bitwise
floating-point equivalence. Compile failures are explicit; disable compilation to
use eager updates. First use for a shape includes compilation overhead; compare
warmed runs when measuring performance. Python changes require a kernel restart
but do not require rebuilding the native extension. Hard MAP inference is unchanged.

Run the execution-control tests from the DeepFlow directory:

```bash
python -m unittest discover -s pytorch -p 'test_binary_lbp_execution.py'
DEEPFLOW_TEST_COMPILE=1 python -m unittest discover -s pytorch -p 'test_binary_lbp_execution.py'
```

The tests use CUDA when available and otherwise CPU. They exercise real PyTorch
LBP/autograd without requiring the native MAP extension.
