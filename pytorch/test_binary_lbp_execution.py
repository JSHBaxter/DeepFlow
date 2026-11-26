"""LBP execution tests; native MAP is deliberately not exercised.

    python -m unittest discover -s pytorch -p 'test_binary_lbp_execution.py'
    DEEPFLOW_TEST_COMPILE=1 python -m unittest discover -s pytorch -p 'test_binary_lbp_execution.py'

When the native extension is unavailable, only its unused import is stubbed. All
solver arithmetic and autograd execute in real PyTorch. CUDA is used if available.
"""
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

# LBP itself does not call the extension. Allow testing it before a native build.
if importlib.util.find_spec("deepflow") is None:
    with patch.dict(sys.modules, {"deepflow": types.ModuleType("deepflow")}):
        import binary_deepflow_function as lbp
        from binary_deepflow import BinaryDeepFlow2dStar
else:
    import binary_deepflow_function as lbp
    from binary_deepflow import BinaryDeepFlow2dStar
from binary_lbp_utils import BinaryLBPOptions, checked_iteration, check_residual


class BinaryLBPExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def problem(self, hard=True):
        d = torch.linspace(-0.6, 0.9, 12, device=self.device).reshape(1, 1, 3, 4)
        r = torch.full_like(d, 0.15)
        lx, ly = torch.zeros_like(d), torch.zeros_like(d)
        if hard:
            lx[:, :, :1, :] = -1
            lx[:, :, 1:-1, :] = 1
            ly[:, :, :, :1] = -1
            ly[:, :, :, 1:-1] = 1
        return d, r, lx, ly

    def run_module(self, compile_updates=False, hard=True, check_every=100):
        d, r, lx, ly = self.problem(hard)
        d, rx, ry = (v.clone().requires_grad_() for v in (d, r, r))
        solver = BinaryDeepFlow2dStar(
            test_type=None, check_every=check_every, compile_updates=compile_updates,
        )
        outputs = solver(d, rx, ry, lx, ly)
        loss = (outputs[0] * torch.arange(12, device=self.device).reshape_as(d)).mean()
        gradients = torch.autograd.grad(loss, (d, rx, ry))
        return outputs[0].detach(), gradients, solver.last_convergence

    def test_check_schedule_and_validation(self):
        options = BinaryLBPOptions()
        checks = [i for i in range(1, 218) if checked_iteration(i, 217, options)]
        self.assertEqual(checks, [1, 100, 200, 217])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                BinaryLBPOptions(check_every=bad)

    def test_residual_metrics_and_original_criteria(self):
        options = BinaryLBPOptions()
        values = [torch.tensor([1., -3.], device=self.device),
                  torch.tensor([2., 2.], device=self.device)]
        trace = {}
        self.assertFalse(check_residual(values, 1, 100, 3., "forward", options, trace))
        self.assertEqual(trace["residual"], 4.)
        self.assertEqual(trace["history"][0]["mean_abs"], 2.)
        self.assertEqual(trace["history"][0]["max_abs"], 3.)
        self.assertTrue(check_residual(values, 100, 100, 3., "forward", options, trace))
        self.assertEqual(trace["status"], "iteration_limit")
        trace = {}
        self.assertTrue(check_residual(values, 1, 100, 3., "backward", options, trace))
        self.assertEqual(trace["status"], "converged")
        self.assertEqual(trace["residual"], 3.)
        trace = {}
        check_residual(values, 1, 100, 3., "forward",
                       BinaryLBPOptions(record_history=False), trace)
        self.assertNotIn("history", trace)
        self.assertEqual(trace["residual"], 4.)

    def test_nonfinite_residual_fails_at_check(self):
        for bad in (float("nan"), float("inf")):
            trace = {}
            with self.assertRaises(FloatingPointError):
                check_residual([torch.tensor([bad], device=self.device)], 1, 10, 1e-7,
                               "forward", BinaryLBPOptions(), trace)
            self.assertEqual(trace["status"], "nonfinite")

    def test_star_autograd_and_histories(self):
        probabilities, gradients, traces = self.run_module()
        self.assertTrue(torch.isfinite(probabilities).all())
        for g in gradients:
            self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(gradients[0].abs().sum().item(), 0)
        for phase in ("forward", "backward"):
            trace = traces[phase]
            iterations = [row["iteration"] for row in trace["history"]]
            self.assertEqual(iterations[0], 1)
            self.assertEqual(iterations[-1], trace["iterations"])
            self.assertNotEqual(trace["status"], "running")
            self.assertTrue(all(i == 1 or i % 100 == 0 or i == trace["iteration_limit"] for i in iterations))
            if trace["status"] == "iteration_limit":
                self.assertEqual(trace["iterations"], trace["iteration_limit"])

    def test_interval_does_not_change_fixed_budget_updates(self):
        d, r, _, _ = self.problem(False)
        outputs = []
        calls = []
        # Hold stopping fixed so this test isolates arithmetic from delayed stopping.
        for interval in (1, 100):
            with patch.object(lbp, "check_residual", return_value=False) as monitor:
                outputs.append(lbp.Binary_LBPnD_PytorchNative_forward(
                    d, [r, r], [r, r], BinaryLBPOptions(check_every=interval),
                ))
                calls.append(monitor.call_count)
        self.assertLess(calls[1], calls[0] / 20)
        self.assertEqual(outputs[0][-1], outputs[1][-1])
        for group_a, group_b in zip(outputs[0][:4], outputs[1][:4]):
            for a, b in zip(group_a, group_b):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_expanded_gradients_and_old_apply_signature(self):
        d, r, _, _ = self.problem(False)
        d = d.requires_grad_()
        messages = lbp.Binary_LBP2d_PytorchNative.apply(d, r, r, r, r)
        sum(message.sum() for message in messages).backward()
        self.assertTrue(torch.isfinite(d.grad).all())

    def test_zero_gradient_fast_exit(self):
        d, r, _, _ = self.problem(False)
        out = lbp.Binary_LBPnD_PytorchNative_forward(d, [r, r], [r, r])
        zeros = [[torch.zeros_like(v) for v in group] for group in out[:4]]
        trace = {}
        gd, gru, grd = lbp.Binary_LBPnD_PytorchNative_backward(
            *zeros, *out[:4], d, [r, r], [r, r], out[-1], BinaryLBPOptions(), trace,
        )
        self.assertEqual(trace["status"], "zero_gradient")
        self.assertEqual(trace["iterations"], 0)
        for value in [gd] + gru + grd:
            self.assertEqual(value.count_nonzero().item(), 0)

    def test_invocations_keep_separate_diagnostics(self):
        d, r, lx, ly = self.problem(False)
        solver = BinaryDeepFlow2dStar(test_type=None)
        first = d.clone().requires_grad_()
        p = solver(first, r, r, lx, ly)[0]
        old_trace = solver.last_convergence
        solver(d, r, r, lx, ly)
        new_trace = solver.last_convergence
        p.sum().backward()
        self.assertEqual(new_trace["backward"]["status"], "not_run")
        self.assertNotEqual(old_trace["backward"]["status"], "not_run")

    def test_other_dimensions_still_backpropagate(self):
        for shape, function in (((1, 1, 4), lbp.Binary_LBP1d_PytorchNative),
                                ((1, 1, 2, 2, 2), lbp.Binary_LBP3d_PytorchNative)):
            d = torch.full(shape, 0.4, device=self.device, requires_grad=True)
            r = torch.full_like(d, 0.1)
            messages = function.apply(d, *([r] * (2 * (len(shape) - 2))))
            sum(message.square().mean() for message in messages).backward()
            self.assertTrue(torch.isfinite(d.grad).all())

    @unittest.skipUnless(os.environ.get("DEEPFLOW_TEST_COMPILE") == "1", "opt-in compilation test")
    def test_compiled_probabilities_and_all_input_gradients(self):
        for hard in (False, True):
            eager = self.run_module(False, hard)
            fused = self.run_module(True, hard)
            torch.testing.assert_close(eager[0], fused[0], rtol=1e-4, atol=1e-6)
            for expected, actual in zip(eager[1], fused[1]):
                torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
