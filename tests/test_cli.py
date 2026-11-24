import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from mizan.cli import main


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    def test_weigh_json(self):
        code, out, _ = self.run_cli("weigh", "--model", "llama-3.1-8b", "--gpu", "8",
                                    "--seq", "512", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["verdict"], "OK")
        self.assertGreater(data["total_gib"], 0)

    def test_weigh_hf_style_id(self):
        code, out, _ = self.run_cli("weigh", "-m", "meta-llama/Llama-3.1-8B-Instruct",
                                    "--gpu", "8", "--seq", "512", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["model"], "llama-3.1-8b")

    def test_weigh_over_capacity(self):
        code, out, _ = self.run_cli("weigh", "-m", "llama-3.1-8b", "--gpu", "4",
                                    "--seq", "512", "--json")
        self.assertEqual(json.loads(out)["verdict"], "OVER")

    def test_fit_returns_candidates(self):
        code, out, _ = self.run_cli("fit", "-m", "llama-3.1-8b", "--gpu", "8",
                                    "--seq", "512", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data["candidates"])
        self.assertIn("batch", data["best"])

    def test_fit_without_gpu_fails_cleanly(self):
        code, _, err = self.run_cli("fit", "-m", "llama-3.1-8b", "--json")
        self.assertEqual(code, 2)
        self.assertIn("no GPU size known", err)

    def test_check_good_and_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps({
                "base": "llama-3.1-8b", "task": "sft",
                "data": {"train": "train.jsonl"}, "training": {"epochs": 1},
            }), encoding="utf-8")
            self.assertEqual(self.run_cli("check", "-c", str(good))[0], 0)

            bad = Path(tmp) / "bad.yaml"
            bad.write_text("base: llama\ntrainin:\n  epochs: 0\n", encoding="utf-8")
            code, out, _ = self.run_cli("check", "-c", str(bad))
            self.assertEqual(code, 1)
            self.assertIn("unknown top-level key", out)

    def test_presets_json(self):
        code, out, _ = self.run_cli("presets", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out))

    def test_doctor_json(self):
        code, out, _ = self.run_cli("doctor", "--json")
        self.assertEqual(code, 0)
        self.assertIn("mizan_version", json.loads(out))

    def test_init_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "mizan.yaml"
            self.assertEqual(self.run_cli("init", "--out", str(cfg))[0], 0)
            self.assertIn("base: llama-3.1-8b", cfg.read_text(encoding="utf-8"))
            report = Path(tmp) / "report.md"
            code, _, _ = self.run_cli("report", "-c", str(cfg), "--seq", "512",
                                      "--out", str(report))
            self.assertEqual(code, 0)
            self.assertIn("## Verdict", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
