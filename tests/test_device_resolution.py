import importlib
import importlib.machinery
import sys
import types
import unittest
from unittest import mock


def install_torch_stub(cuda_available: bool) -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub


class ResolveDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_torch = sys.modules.get("torch")
        install_torch_stub(cuda_available=False)
        import prepare

        self.prepare = importlib.reload(prepare)

    def tearDown(self) -> None:
        if self._previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._previous_torch

    def test_auto_prefers_cuda_when_available(self) -> None:
        install_torch_stub(cuda_available=True)
        module = importlib.reload(self.prepare)

        self.assertEqual(module.resolve_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu_without_cuda(self) -> None:
        self.assertEqual(self.prepare.resolve_device("auto"), "cpu")

    def test_explicit_cuda_raises_without_gpu(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no CUDA-capable device"):
            self.prepare.resolve_device("cuda")

    def test_explicit_cpu_is_preserved(self) -> None:
        self.assertEqual(self.prepare.resolve_device("cpu"), "cpu")

    def test_explicit_mps_requires_available_backend(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no Apple Metal device"):
            self.prepare.resolve_device("mps")
        sys.modules["torch"].backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        )
        self.assertEqual(self.prepare.resolve_device("mps"), "mps")
        self.assertTrue(self.prepare.collect_environment_report()["mps_available"])


class CliDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_torch = sys.modules.get("torch")
        install_torch_stub(cuda_available=False)

    def tearDown(self) -> None:
        if self._previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._previous_torch

    def test_bench_defaults_to_auto_device(self) -> None:
        import bench

        module = importlib.reload(bench)
        with mock.patch.object(sys, "argv", ["bench.py"]):
            args = module.parse_args()

        self.assertEqual(args.device, "auto")
        self.assertIsNone(args.iterations)
        self.assertIsNone(args.warmup)

    def test_profile_defaults_to_auto_device(self) -> None:
        import profile

        module = importlib.reload(profile)
        with mock.patch.object(sys, "argv", ["profile.py"]):
            args = module.parse_args()

        self.assertEqual(args.device, "auto")


if __name__ == "__main__":
    unittest.main()
