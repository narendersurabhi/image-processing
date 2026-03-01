import unittest

try:
    from portrait_enhancer.core.utils import resolve_acceleration_mode

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "opencv deps not installed")
class AccelSmokeTests(unittest.TestCase):
    def test_resolve_acceleration_mode_returns_known_mode(self):
        self.assertIn(resolve_acceleration_mode("auto"), ("cpu", "cuda"))
        self.assertEqual(resolve_acceleration_mode("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
