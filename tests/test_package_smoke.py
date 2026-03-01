import importlib
import unittest


class PackageSmokeTests(unittest.TestCase):
    def test_import_package_without_booting_ui(self):
        pkg = importlib.import_module("portrait_enhancer")
        self.assertEqual(pkg.__version__, "0.1.0")
        self.assertTrue(callable(pkg.run_app))
        self.assertTrue(callable(pkg.create_app))


if __name__ == "__main__":
    unittest.main()
