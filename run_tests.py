import sys
import os
import pytest
from pathlib import Path

def main():
    root = Path(__file__).parent
    # Add module dirs
    sys.path.insert(0, str(root / "ForceSMIP"))
    sys.path.insert(0, str(root / "cope_methods"))
    test_dir = root / "tests"
    if not test_dir.exists():
        print("No tests/ directory found.")
        return 1
    # Run pytest programmatically
    args = [
        str(test_dir),
        "-q",
        "--disable-warnings",
        "--maxfail=1",
    ]
    print("Running pytest with args:", " ".join(args))
    return pytest.main(args)

if __name__ == "__main__":
    raise SystemExit(main())