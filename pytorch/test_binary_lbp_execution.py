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
from binary_lbp_utils import (BinaryLBPOptions, checked_iteration, check_residual,
                              forward_step, node_probabilities)


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

    def run_module(self, compile_updates=False, hard=True, check_every=100, monitor_updates=True):
        d, r, lx, ly = self.problem(hard)
        d, rx, ry = (v.clone().requires_grad_() for v in (d, r, r))
        solver = BinaryDeepFlow2dStar(
            test_type=None, check_every=check_every, compile_updates=compile_updates,
            monitor_updates=monitor_updates,
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

    def stable_metrics(self, update=0., message=0., probability=0.):
        return {"actual_update_max_abs": torch.tensor(update, device=self.device),
                "actual_update_mean_abs": torch.tensor(update, device=self.device),
                "message_change_max_abs": None if message is None else torch.tensor(message, device=self.device),
                "probability_change_max_abs": None if probability is None else torch.tensor(probability, device=self.device)}

    def test_stall_requires_multiple_checks_and_never_claims_convergence(self):
        options = BinaryLBPOptions(stop_on_stall=True, stall_patience=2, stall_min_iterations=0)
        trace = {}
        values = [torch.ones(2, device=self.device)]
        self.assertFalse(check_residual(values, 1, 500, 1e-7, "forward", options, trace,
                                        self.stable_metrics(message=None, probability=None)))
        self.assertEqual(trace["stable_checks"], 0)
        self.assertFalse(check_residual(values, 100, 500, 1e-7, "forward", options, trace, self.stable_metrics()))
        self.assertTrue(check_residual(values, 200, 500, 1e-7, "forward", options, trace, self.stable_metrics()))
        self.assertEqual(trace["status"], "stalled")
        self.assertEqual(trace["stagnation_reason"], "exact_freeze")
        self.assertFalse(trace["converged"])

    def test_stall_is_diagnostic_only_by_default_without_history(self):
        options = BinaryLBPOptions(stall_patience=1, stall_min_iterations=0, record_history=False)
        trace = {}
        self.assertFalse(check_residual([torch.ones(1, device=self.device)], 100, 500, 1e-7,
                                        "forward", options, trace, self.stable_metrics()))
        self.assertTrue(trace["stagnation_detected"])
        self.assertEqual(trace["status"], "running")
        self.assertNotIn("history", trace)

    def test_flat_proposal_residual_does_not_imply_stall(self):
        options = BinaryLBPOptions(stop_on_stall=True, stall_patience=1, stall_min_iterations=0)
        for changed in ("actual_update_max_abs", "message_change_max_abs", "probability_change_max_abs"):
            trace = {}
            metrics = self.stable_metrics()
            metrics[changed] = torch.tensor(1e-6, device=self.device)
            self.assertFalse(check_residual([torch.ones(1, device=self.device)], 100, 500, 1e-7,
                                            "forward", options, trace, metrics))
            self.assertFalse(trace["stagnation_detected"])

    def test_stability_counter_resets_and_honours_minimum_iterations(self):
        options = BinaryLBPOptions(stop_on_stall=True, stall_patience=2, stall_min_iterations=200)
        trace = {}
        values = [torch.ones(1, device=self.device)]
        check_residual(values, 100, 600, 1e-7, "forward", options, trace, self.stable_metrics())
        self.assertEqual(trace["stable_checks"], 0)
        check_residual(values, 200, 600, 1e-7, "forward", options, trace, self.stable_metrics())
        self.assertEqual(trace["stable_checks"], 1)
        check_residual(values, 300, 600, 1e-7, "forward", options, trace, self.stable_metrics(message=1e-6))
        self.assertEqual(trace["stable_checks"], 0)

    def test_monitoring_leaves_eager_probabilities_and_gradients_unchanged(self):
        monitored = self.run_module(monitor_updates=True)
        unmonitored = self.run_module(monitor_updates=False)
        torch.testing.assert_close(monitored[0], unmonitored[0], rtol=0, atol=0)
        for a, b in zip(monitored[1], unmonitored[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        trace = monitored[2]["forward"]
        self.assertIsNone(trace["history"][0]["probability_change_max_abs"])
        self.assertEqual(trace["probability_min"], monitored[0].min().item())
        self.assertEqual(trace["probability_max"], monitored[0].max().item())
        self.assertNotIn("actual_update_max_abs", unmonitored[2]["forward"])
        # Traces are JSON-compatible and retain no GPU tensors.
        import json
        json.dumps(monitored[2], allow_nan=False)

    def test_actual_update_metric_is_after_damping(self):
        d, r, _, _ = self.problem(False)
        groups = [[torch.ones((1, 1, 2, 4), device=self.device),
                   torch.ones((1, 1, 3, 3), device=self.device)] for _ in range(4)]
        before = [v.clone() for g in (groups[1], groups[3], groups[0], groups[2]) for v in g]
        result = forward_step(lbp.Binary_LBPnD_PytorchNative_forward_compute_vertex_messages,
                              torch.exp(-d * (d > 0)), torch.exp(d * (d < 0)),
                              [torch.exp(-r)] * 2, [torch.exp(-r)] * 2, *groups, True, True)
        after = [v for g in (result[1], result[3], result[0], result[2]) for v in g]
        for update, old, new in zip(result[5], before, after):
            torch.testing.assert_close(update, new-old, rtol=0, atol=0)
        self.assertGreater(result[4][0].abs().max().item(), result[5][0].abs().max().item())

    def test_node_probability_monitor_uses_layer_arithmetic(self):
        d, r, _, _ = self.problem(False)
        result = lbp.Binary_LBPnD_PytorchNative_forward(
            d, [r, r], [r, r], BinaryLBPOptions(monitor_updates=False))
        probabilities = node_probabilities(torch.exp(-d * (d > 0)), torch.exp(d * (d < 0)), *result[:4])
        zeros = torch.zeros_like(d)
        expected = BinaryDeepFlow2dStar(test_type=None)(d, r, r, zeros, zeros)[0]
        torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)

    def test_real_float32_freeze_preserves_outputs_gradients_and_adjoint_budget(self):
        # A deterministic CPU example demonstrates the proposal/update distinction.
        # Test with actual LBP rather than a mocked stopping decision.
        d = torch.full((1, 1, 3, 4), 0.5)
        r = torch.full_like(d, 0.15)
        zero = torch.zeros_like(d)
        results = []
        for stop in (False, True):
            x, rx, ry = (value.clone().requires_grad_() for value in (d, r, r))
            solver = BinaryDeepFlow2dStar(test_type=None, check_every=50,
                                         stop_on_stall=stop, stall_min_iterations=0, stall_patience=2)
            p = solver(x, rx, ry, zero, zero)[0]
            gradients = torch.autograd.grad(p.square().mean(), (x, rx, ry))
            results.append((p.detach(), gradients, solver.last_convergence))
        baseline, stopped = results
        self.assertEqual(baseline[2]["forward"]["status"], "iteration_limit")
        self.assertEqual(stopped[2]["forward"]["status"], "stalled")
        self.assertGreater(stopped[2]["forward"]["residual"], 1e-7)
        self.assertLess(stopped[2]["forward"]["iterations"], baseline[2]["forward"]["iterations"])
        self.assertEqual(stopped[2]["backward"]["iteration_limit"], baseline[2]["backward"]["iteration_limit"])
        torch.testing.assert_close(stopped[0], baseline[0], rtol=0, atol=0)
        for actual, expected in zip(stopped[1], baseline[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_stall_option_validation(self):
        for kwargs in ({"stop_on_stall": True, "monitor_updates": False},
                       {"stall_patience": 0}, {"stall_min_iterations": -1},
                       {"stall_message_atol": -1}, {"stall_probability_atol": float("nan")}):
            with self.assertRaises(ValueError):
                BinaryLBPOptions(**kwargs)

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
