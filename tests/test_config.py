"""Tests for configuration loading, validation, and override precedence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lume.config import Config, config_path, data_home, load_config, save_config


class TestPaths(unittest.TestCase):
    def test_explicit_env_wins(self):
        self.assertEqual(config_path({"LUME_CONFIG": "/tmp/x.json"}), Path("/tmp/x.json"))
        self.assertEqual(data_home({"LUME_HOME": "/tmp/lume"}), Path("/tmp/lume"))

    def test_xdg_paths(self):
        self.assertEqual(config_path({"XDG_CONFIG_HOME": "/c"}), Path("/c/lume/config.json"))
        self.assertEqual(data_home({"XDG_DATA_HOME": "/d"}), Path("/d/lume"))

    def test_defaults_are_under_home(self):
        self.assertTrue(str(config_path({})).endswith("/.config/lume/config.json")
                        or "lume" in str(config_path({})))


class TestValidation(unittest.TestCase):
    def test_bad_values_are_clamped_with_warnings(self):
        cfg = Config(theme="nope", effort="turbo", max_tokens=99999999,
                     max_retries=-4, timeout="soon").validate()
        self.assertEqual(cfg.theme, "auto")
        self.assertEqual(cfg.effort, "high")
        self.assertEqual(cfg.max_tokens, 128000)
        self.assertEqual(cfg.max_retries, 0)
        self.assertEqual(cfg.timeout, 600.0)
        self.assertGreaterEqual(len(cfg.warnings), 4)

    def test_empty_model_reverts_to_default(self):
        self.assertEqual(Config(model="  ").validate().model, "claude-opus-5")

    def test_defaults_are_valid(self):
        cfg = Config().validate()
        self.assertEqual(cfg.warnings, [])
        self.assertEqual(cfg.model, "claude-opus-5")

    def test_width_for_respects_cap_and_floor(self):
        cfg = Config(max_width=100).validate()
        self.assertEqual(cfg.width_for(200), 100)
        self.assertEqual(cfg.width_for(64), 64)
        self.assertEqual(cfg.width_for(3), 20)
        self.assertEqual(Config(max_width=0).validate().width_for(200), 200)


class TestLoadSave(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "config.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file_gives_defaults(self):
        cfg = load_config(self.path, env={})
        self.assertEqual(cfg.model, "claude-opus-5")
        self.assertEqual(cfg.warnings, [])

    def test_round_trip(self):
        cfg = Config(model="claude-sonnet-5", theme="ember", max_tokens=4096).validate()
        save_config(cfg, self.path)
        again = load_config(self.path, env={})
        self.assertEqual(again.model, "claude-sonnet-5")
        self.assertEqual(again.theme, "ember")
        self.assertEqual(again.max_tokens, 4096)

    def test_saved_file_is_private(self):
        save_config(Config(), self.path)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_malformed_json_does_not_raise(self):
        self.path.write_text("{ not json", encoding="utf-8")
        cfg = load_config(self.path, env={})
        self.assertEqual(cfg.model, "claude-opus-5")
        self.assertTrue(cfg.warnings)

    def test_non_object_root_does_not_raise(self):
        self.path.write_text("[1,2,3]", encoding="utf-8")
        self.assertTrue(load_config(self.path, env={}).warnings)

    def test_unknown_keys_are_preserved_not_lost(self):
        self.path.write_text(json.dumps({"model": "claude-sonnet-5", "future_flag": 7}),
                             encoding="utf-8")
        cfg = load_config(self.path, env={})
        self.assertEqual(cfg.unknown["future_flag"], 7)
        save_config(cfg, self.path)
        self.assertEqual(json.loads(self.path.read_text())["future_flag"], 7)

    def test_no_temp_file_left_behind(self):
        save_config(Config(), self.path)
        self.assertEqual([p.name for p in self.path.parent.glob("*.tmp")], [])


class TestOverrides(unittest.TestCase):
    def test_env_overrides_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            save_config(Config(model="claude-sonnet-5").validate(), p)
            cfg = load_config(p, env={"LUME_MODEL": "claude-haiku-4-5", "LUME_THEME": "ember"})
            self.assertEqual(cfg.model, "claude-haiku-4-5")
            self.assertEqual(cfg.theme, "ember")

    def test_env_no_motion(self):
        self.assertFalse(load_config(Path("/nonexistent/x.json"),
                                     env={"LUME_NO_MOTION": "1"}).animation)

    def test_cli_overrides_win_and_none_is_ignored(self):
        cfg = Config(model="claude-sonnet-5").validate()
        cfg.apply_overrides(model=None, theme="ember", effort="max")
        self.assertEqual(cfg.model, "claude-sonnet-5")
        self.assertEqual(cfg.theme, "ember")
        self.assertEqual(cfg.effort, "max")

    def test_overrides_are_validated(self):
        cfg = Config().validate()
        cfg.apply_overrides(theme="bogus")
        self.assertEqual(cfg.theme, "auto")

    def test_unknown_override_key_is_ignored(self):
        Config().validate().apply_overrides(not_a_field=1)  # must not raise




class TestThemeAuto(unittest.TestCase):
    """The default theme follows the terminal rather than assuming it is dark."""

    def test_the_default_is_auto(self):
        self.assertEqual(Config().validate().theme, "auto")

    def test_an_explicit_theme_is_kept(self):
        for name in ("aurora", "solar", "ember", "mono"):
            self.assertEqual(Config(theme=name).validate().theme, name)

    def test_case_and_whitespace_are_normalised(self):
        self.assertEqual(Config(theme="  EmBeR ").validate().theme, "ember")

    def test_an_unknown_theme_falls_back_to_auto_with_a_warning(self):
        cfg = Config(theme="neon").validate()
        self.assertEqual(cfg.theme, "auto")
        self.assertTrue(cfg.warnings)


class TestXdgPathsAgreeWithTheStore(unittest.TestCase):
    """Two modules resolving one rule is how history and sessions drifted apart."""

    def test_a_relative_xdg_data_home_is_ignored_by_both(self):
        from lume.store import default_root
        env = {"XDG_DATA_HOME": "relative/path"}
        self.assertEqual(str(data_home(env)), str(default_root(env)))
        self.assertTrue(data_home(env).is_absolute())

    def test_an_absolute_xdg_data_home_is_honoured_by_both(self):
        from lume.store import default_root
        env = {"XDG_DATA_HOME": "/tmp/xdg-abs"}
        self.assertEqual(str(data_home(env)), str(default_root(env)))

    def test_lume_home_still_wins_for_both(self):
        from lume.store import default_root
        env = {"LUME_HOME": "/tmp/lh", "XDG_DATA_HOME": "/tmp/xdg-abs"}
        self.assertEqual(str(data_home(env)), str(default_root(env)))

    def test_a_relative_xdg_config_home_is_ignored(self):
        path = config_path({"XDG_CONFIG_HOME": "relative/conf"})
        self.assertTrue(path.is_absolute())
        self.assertNotIn("relative", str(path))

    def test_an_absolute_xdg_config_home_is_honoured(self):
        self.assertEqual(config_path({"XDG_CONFIG_HOME": "/tmp/c"}),
                         Path("/tmp/c/lume/config.json"))

if __name__ == "__main__":
    unittest.main()
