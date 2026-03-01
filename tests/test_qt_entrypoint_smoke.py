import unittest

import portrait_enhancer


class QtEntrypointSmokeTests(unittest.TestCase):
    def test_qt_entrypoint_is_exposed_without_importing_qt(self):
        self.assertTrue(callable(portrait_enhancer.run_qt_app))
        self.assertTrue(callable(portrait_enhancer.create_qt_app))


if __name__ == "__main__":
    unittest.main()
