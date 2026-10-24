"""Retain unit-test evidence; this does not run robot physics."""
import hashlib
import io
import json
from pathlib import Path
import unittest

from laundry_task import test_supervisor


def main():
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_supervisor)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "scope": "Pure-Python task protocol tests with synthetic observations; not physical laundry results",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "passed": result.wasSuccessful(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("supervisor.py", "test_supervisor.py", "validate_protocol.py")},
        "test_output": stream.getvalue(),
    }
    (root / "output").mkdir(exist_ok=True)
    (root / "output/protocol_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
