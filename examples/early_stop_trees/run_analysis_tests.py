#!/usr/bin/env python
"""Run the focused inference tests without depending on a pytest launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    test_path = Path(__file__).with_name("test_analysis_inference.py")
    spec = importlib.util.spec_from_file_location("test_analysis_inference", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {test_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tests = sorted(
        (name, value)
        for name, value in vars(module).items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} focused inference tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
