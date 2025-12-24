#!/usr/bin/env python3
"""
Test Runner CLI for KI Trading Model

Usage:
    python run_tests.py                  # Interactive mode, localhost
    python run_tests.py -H 10.1.19.101   # Interactive mode, remote host
    python run_tests.py -n               # Batch mode (no interaction)
    python run_tests.py --pytest         # Run pytest instead
"""

import subprocess
import sys
import os

def main():
    # Check if pytest mode requested
    if "--pytest" in sys.argv or "-p" in sys.argv:
        # Run standard pytest
        args = [sys.executable, "-m", "pytest", "tests/", "-v"]

        # Pass through host if specified
        for i, arg in enumerate(sys.argv):
            if arg in ["-H", "--host"] and i + 1 < len(sys.argv):
                os.environ["TEST_SERVICE_HOST"] = sys.argv[i + 1]

        print("Running pytest suite...")
        sys.exit(subprocess.call(args))

    # Run interactive runner
    script_path = os.path.join(os.path.dirname(__file__), "tests", "interactive_runner.py")

    # Build args, passing through CLI arguments
    args = [sys.executable, script_path]

    # Forward all arguments except the script name
    for arg in sys.argv[1:]:
        args.append(arg)

    sys.exit(subprocess.call(args))


if __name__ == "__main__":
    main()
