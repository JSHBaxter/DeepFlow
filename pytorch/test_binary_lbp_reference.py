"""Float64 buffer, autograd and frozen-problem reference tests (native MAP unused)."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

if importlib.util.find_spec("deepflow") is None:
    with patch.dict(sys.modules, {"deepflow": types.ModuleType("deepflow")}):
        import binary_deepflow_function as lbp
        from binary_deepflow import BinaryDeepFlow2dStar
        from binary_lbp_reference import compare_binary_star_precision
else:
    import binary_deepflow_function as lbp
    from binary_deepflow import BinaryDeepFlow2dStar
    from binary_lbp_reference import compare_binary_star_precision
from binary_lbp_utils import BinaryLBPOptions


class BinaryLBPReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def problem(self, device=None):
        device = device or self.device
        d = torch.full((1, 1, 3, 4), 0.5, device=device)
        r = torch.full_like(d, .15)
        zero = torch.zeros_like(d)
        target = torch.zeros_like(d)
        target[:, :, 1:, 1:3] = 1
        return (d, r, r.clone(), zero, zero.clone()), target

    def test_all_forward_and_backward_buffers_inherit_float64(self):
        inputs, _ = self.problem()
        d, rx, ry, lx, ly = (value.double() for value in inputs)
        d.requires_grad_(); rx.requires_grad_(); ry.requires_grad_()
        forward_step, backward_step = lbp.forward_step, lbp.backward_step
        seen = {"forward": 0, "backward": 0}

        def inspect_forward(vertex, ed0, ed1, eru, erd, *args):
            seen["forward"] += 1
            for group in (eru, erd, *args[:4]):
                self.assertTrue(all(value.dtype == torch.float64 for value in group))
            self.assertEqual(ed0.dtype, torch.float64)
            self.assertEqual(ed1.dtype, torch.float64)
            return forward_step(vertex, ed0, ed1, eru, erd, *args)

        def inspect_backward(*args):
            seen["backward"] += 1
            for arg in args:
                for value in arg if isinstance(arg, list) else (arg,):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        self.assertEqual(value.dtype, torch.float64)
            return backward_step(*args)

        with patch.object(lbp, "forward_step", inspect_forward), patch.object(lbp, "backward_step", inspect_backward):
            solver = BinaryDeepFlow2dStar(test_type=None)
            probabilities = solver(d, rx, ry, lx, ly)[0]
            gradients = torch.autograd.grad(probabilities.square().mean(), (d, rx, ry))
        self.assertGreater(seen["forward"], 0)
        self.assertGreater(seen["backward"], 0)
        self.assertEqual(probabilities.dtype, torch.float64)
        self.assertTrue(all(g.dtype == torch.float64 and torch.isfinite(g).all() for g in gradients))

    def test_float64_continues_past_float32_rounding_floor(self):
        # CPU makes this rounding example reproducible across test machines.
        inputs, targets = self.problem(torch.device("cpu"))
        with contextlib.redirect_stdout(io.StringIO()):
            report, artifacts = compare_binary_star_precision(inputs, targets, check_every=50)
        f32 = report["float32"]["trace"]["forward"]
        f64 = report["float64"]["trace"]["forward"]
        # The tiny problem's cap is below the detector's 1000-step minimum;
        # verify the actual freeze rather than expecting its patience flag.
        self.assertEqual(f32["actual_update_max_abs"], 0.)
        self.assertEqual(f32["message_change_max_abs"], 0.)
        self.assertEqual(f32["status"], "iteration_limit")
        self.assertEqual(f64["status"], "converged")
        self.assertLess(f64["residual"], 1e-7)
        self.assertTrue(report["float64_converged"])
        self.assertEqual(artifacts["float64"]["probabilities"].dtype, torch.float64)
        self.assertEqual(artifacts["float64"]["unary_gradient"].dtype, torch.float64)
        json.dumps(report, allow_nan=False)

    def test_reference_preserves_inputs_gradients_and_rng(self):
        inputs, target = self.problem()
        inputs[0].requires_grad_()
        copies = [value.detach().clone() for value in inputs]
        state = torch.random.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
            report, artifacts = compare_binary_star_precision(inputs, target, metadata={"source": "test"})
        for actual, expected in zip(inputs, copies):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(inputs[0].grad)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(report["metadata"], {"source": "test"})
        for value in artifacts["inputs"].values():
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)

    def test_invalid_frozen_problem(self):
        inputs, target = self.problem()
        with self.assertRaises(TypeError):
            compare_binary_star_precision(tuple(t.double() for t in inputs), target)
        with self.assertRaises(ValueError):
            compare_binary_star_precision(inputs, target[:, :, :-1])
        with self.assertRaises(ValueError):
            compare_binary_star_precision(inputs, target, smooth=float("nan"))

    @unittest.skipUnless(os.environ.get("DEEPFLOW_TEST_COMPILE") == "1", "opt-in float64 compilation")
    def test_compiled_float64_probability_and_gradient_parity(self):
        inputs, _ = self.problem()
        results = []
        for compiled in (False, True):
            d, rx, ry, lx, ly = (value.double().clone() for value in inputs)
            d.requires_grad_()
            solver = BinaryDeepFlow2dStar(test_type=None, compile_updates=compiled)
            p = solver(d, rx, ry, lx, ly)[0]
            gradient = torch.autograd.grad(p.square().mean(), d)[0]
            results.append((p.detach(), gradient))
        for expected, actual in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
