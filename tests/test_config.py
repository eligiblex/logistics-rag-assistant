import os
import unittest
from unittest.mock import patch

from src.config import ConfigManager


class ConfigTests(unittest.TestCase):
    def test_environment_overrides_yaml(self):
        with patch.dict(
            os.environ,
            {"OLLAMA_BASE_URL": "http://ollama:11434", "SERVER_PORT": "9000"},
        ):
            config = ConfigManager("missing-config.yaml").get_config()
        self.assertEqual(config.ollama.base_url, "http://ollama:11434")
        self.assertEqual(config.server.port, 9000)


if __name__ == "__main__":
    unittest.main()
