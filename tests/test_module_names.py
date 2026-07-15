import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ModuleNameTests(unittest.TestCase):
    def test_expected_module_files_exist(self):
        self.assertTrue((ROOT / "events.py").exists(), "events.py should exist")
        self.assertTrue((ROOT / "intent_classifier.py").exists(), "intent_classifier.py should exist")

    def test_modules_can_be_loaded(self):
        for module_name in ["events", "intent_classifier"]:
            module_path = ROOT / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)


if __name__ == "__main__":
    unittest.main()
