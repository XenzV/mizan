# Contributing

Run the tests before opening a PR:

```bash
pip install -e ".[dev]"
pytest -q
python -m unittest discover -s tests -q
```

Keep the CLI dependency-free. New features should work without PyTorch, CUDA, or network access.

The numbers in `src/mizan/estimator.py` are assumptions, not doctrine. If you measured a real peak that disagrees with Mizan, open an issue using the **measurement** template with the model, config, GPU, estimated value, and measured value. A correction backed by numbers is always welcome.

Add a test for any changed formula or new lint rule. Update `docs/METHOD.md` when a constant changes.

Keep commits focused and describe what changed and why.
