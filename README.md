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
    (1) Marginal probability estimation via iterative mean-field approximations
        which happens to be well-behaved and differentiable provided that the
        regularisation terms are kept within a particular bound, and
    (2) Maximum a posteriori probability estimation via an augmented Lagrangian
        solution algorithm [2], which is not differentiable but can be used for
        layer evaluation (but not for training). This solver works best if the data
        terms are normalized into a -1 to 1 range, but is mathematically scale-
        invariant.

The former solver is based on that presented in 
[1] Baxter, J.S.H., Jannin, P. (2020) Topology-aware activation layer for neural network image segmentation. Proceedings of SPIE Medical Imaging, vol. 11313 p. 11313A. 

The latter solver is based on that presented in:
[2] Baxter, J.S.H., Rajchl, M., Yuan, J., & Peters, T. M. (2014). A continuous max-flow approach to multi-labeling problems under arbitrary region regularization. arXiv preprint arXiv:1405.0892.

At the moment, the project is designed for Linux and make. Depending on the
NVCC compiler used for the CUDA portions, one may need to set the corresponding
C compiler to a particular version using the C make variable.

