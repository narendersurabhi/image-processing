import unittest

try:
    import numpy as np
    from portrait_enhancer.ui.app import PortraitEnhancerV2

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/tkinter deps not installed")
class ProjectStateSmokeTests(unittest.TestCase):
    def test_array_and_masks_round_trip(self):
        app = PortraitEnhancerV2()
        app.withdraw()
        try:
            arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
            masks = {
                "face": np.ones((4, 4), dtype=np.float32),
                "skin": np.zeros((4, 4), dtype=np.float32),
            }
            encoded_arr = app._encode_array(arr)
            decoded_arr = app._decode_array(encoded_arr)
            self.assertTrue(np.array_equal(arr, decoded_arr))

            encoded_masks = app._encode_masks(masks)
            decoded_masks = app._decode_masks(encoded_masks)
            self.assertTrue(np.array_equal(masks["face"], decoded_masks["face"]))
            self.assertTrue(np.array_equal(masks["skin"], decoded_masks["skin"]))
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
