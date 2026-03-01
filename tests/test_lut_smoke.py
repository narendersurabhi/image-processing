import os
import tempfile
import unittest

try:
    import numpy as np

    from portrait_enhancer.core.lut import apply_cube_lut, load_cube_lut

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy deps not installed")
class LutSmokeTests(unittest.TestCase):
    def test_identity_cube_lut_round_trip(self):
        cube = """TITLE "Identity"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
0.0 0.0 1.0
0.0 1.0 0.0
0.0 1.0 1.0
1.0 0.0 0.0
1.0 0.0 1.0
1.0 1.0 0.0
1.0 1.0 1.0
"""
        with tempfile.NamedTemporaryFile("w", suffix=".cube", delete=False) as fh:
            fh.write(cube)
            path = fh.name
        try:
            lut = load_cube_lut(path)
            img = np.array([[[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]]], dtype=np.float32)
            out = apply_cube_lut(img, lut)
            self.assertTrue(np.allclose(img, out, atol=1e-5))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
