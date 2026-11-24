import json
import tempfile
import unittest
from pathlib import Path

from mizan.config import _mini_yaml, lint_config, load_config


YAML_TEXT = """
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: alpaca
training:
  epochs: 3
  lr: 2e-5
  batch_size: 1
  gradient_accumulation: 16
  quantization: 4bit
  gradient_checkpointing: true
  lora:
    r: 16
    target_modules: [q, k, v, o, gate, up, down]
mizan:
  gpu_gib: 8
"""

BAD_YAML = """
base: llama-3.1-8b
trainin:
  epochs: 0
data: {}
"""


class MiniYamlTests(unittest.TestCase):
    def test_nested_mapping_and_inline_list(self):
        data = _mini_yaml(YAML_TEXT)
        self.assertEqual(data["training"]["lora"]["r"], 16)
        self.assertEqual(data["training"]["lora"]["target_modules"][0], "q")
        self.assertEqual(data["mizan"]["gpu_gib"], 8)
        self.assertIs(data["training"]["gradient_checkpointing"], True)
        self.assertEqual(data["training"]["lr"], 2e-5)

    def test_none_stays_a_string_for_quantization(self):
        data = _mini_yaml("training:\n  quantization: none\n")
        self.assertEqual(data["training"]["quantization"], "none")


class LoadTests(unittest.TestCase):
    def test_json_and_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.json").write_text(json.dumps({"base": "x"}), encoding="utf-8")
            (d / "a.toml").write_text('base = "x"\n', encoding="utf-8")
            (d / "a.yaml").write_text("base: x\n", encoding="utf-8")
            for name in ("a.json", "a.toml", "a.yaml"):
                self.assertEqual(load_config(d / name)["base"], "x")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config("does-not-exist.yaml")


class LintTests(unittest.TestCase):
    def test_good_config_has_no_errors(self):
        issues = lint_config(_mini_yaml(YAML_TEXT))
        self.assertEqual([i for i in issues if i.level == "error"], [])

    def test_bad_config_reports_unknown_key_and_nonpositive_epochs(self):
        issues = lint_config(_mini_yaml(BAD_YAML))
        paths = {i.path for i in issues if i.level == "error"}
        self.assertIn("trainin", paths)
        self.assertIn("data.train", paths)

    def test_strict_promotes_warnings(self):
        cfg = _mini_yaml(YAML_TEXT)
        cfg["training"]["lora"]["r"] = 512
        normal = lint_config(cfg)
        strict = lint_config(cfg, strict=True)
        self.assertTrue(any(i.level == "warning" for i in normal))
        self.assertTrue(all(i.level == "error" for i in strict))

    def test_unknown_quantization_is_an_error(self):
        cfg = _mini_yaml(YAML_TEXT)
        cfg["training"]["quantization"] = "3bit"
        self.assertTrue(any(i.path == "training.quantization" for i in lint_config(cfg)))


if __name__ == "__main__":
    unittest.main()
