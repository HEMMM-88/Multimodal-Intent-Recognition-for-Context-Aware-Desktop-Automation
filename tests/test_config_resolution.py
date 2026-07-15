import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main


class ConfigResolutionTests(unittest.TestCase):
    def test_load_config_resolves_relative_to_project_dir(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                data = main.load_config("config.yaml")
            finally:
                os.chdir(original_cwd)

        self.assertIsInstance(data, dict)
        self.assertIn("settings", data)


if __name__ == "__main__":
    unittest.main()
