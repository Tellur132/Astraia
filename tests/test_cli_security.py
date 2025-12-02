from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from astraia import cli


class EnvKeyTests(TestCase):
    def setUp(self) -> None:
        self._saved_env: dict[str, str | None] = {}
        for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "OPENAI_ORG_ID"]:
            self._saved_env[key] = os.environ.get(key)
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_ensure_env_keys_loads_values_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAI_API_KEY=secret-key\n")

            values = cli.ensure_env_keys(
                {"provider": "openai", "model": "gpt-4o"}, env_path=env_path
            )

            self.assertEqual(values["OPENAI_API_KEY"], "secret-key")
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "secret-key")

    def test_ensure_env_keys_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            with self.assertRaises(SystemExit):
                cli.ensure_env_keys({"provider": "openai", "model": "gpt-4o"}, env_path=env_path)

    def test_ensure_env_keys_requires_non_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAI_API_KEY=\n")

            with self.assertRaises(SystemExit):
                cli.ensure_env_keys({"provider": "openai", "model": "gpt-4o"}, env_path=env_path)

    def test_ping_llm_provider_invokes_ping_method(self) -> None:
        provider = MagicMock()
        provider.ping = MagicMock()

        with patch("astraia.cli.create_llm_provider", return_value=(provider, None, None)) as mocked:
            cli.ping_llm_provider({"provider": "openai", "model": "gpt-4o"})

        mocked.assert_called_once()
        provider.ping.assert_called_once()

    def test_ping_llm_provider_errors_when_provider_missing(self) -> None:
        with patch("astraia.cli.create_llm_provider", return_value=(None, None, None)):
            with self.assertRaises(SystemExit):
                cli.ping_llm_provider({"provider": "openai", "model": "gpt-4o"})
